import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Conv2(Conv):

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse

def Correction(x1, x2):
    diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
    diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
    return x1

class Up(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return x1

class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


def Upsample(x, size, align_corners = False):
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                      groups=hidden_features),
            act_layer()
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_shortcut = x
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PSA_Attention(nn.Module):
    def __init__(self, dim, num_heads=8,
                 attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (
            (q.transpose(-2, -1) @ k) * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

class MHSA_CGLU(nn.Module):
    def __init__(self,
                 inc,
                 drop_path=0.1,
                 ):
        super().__init__()
        self.norm1 = LayerNorm2d(inc)
        self.norm2 = LayerNorm2d(inc)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = ConvolutionalGLU(inc)
        self.mhsa = PSA_Attention(inc, num_heads=8)

    def forward(self, x):
        shortcut = x
        x = self.drop_path(self.mhsa(self.norm1(x))) + shortcut
        x = self.drop_path(self.mlp(self.norm2(x))) + x
        return x

class Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class PartiallyTransformerBlock(nn.Module):
    def __init__(self, c, tcr, shortcut=True) -> None:
        super().__init__()
        self.t_ch = int(c * tcr)
        self.c_ch = c - self.t_ch

        self.c_b = Bottleneck(self.c_ch, self.c_ch, shortcut=shortcut)
        self.maxpool = nn.MaxPool2d(2)
        self.t_b = MHSA_CGLU(self.t_ch)

        self.conv_fuse = Conv(c, c)

    def forward(self, x):
        cnn_branch, transformer_branch = x.split((self.c_ch, self.t_ch), 1)

        cnn_branch = self.c_b(cnn_branch)
        transformer_branch = self.t_b(transformer_branch)
        return self.conv_fuse(torch.cat([cnn_branch, transformer_branch], dim=1))


class DBTB(nn.Module):

    def __init__(self, c1, c2, n=1, tcr=0.25, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(PartiallyTransformerBlock(self.c, tcr, shortcut=shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class ConditionalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, context_channels, stride=1):
        super(ConditionalConv2d, self).__init__()
        self.context_conv = nn.Conv2d(context_channels, in_channels * kernel_size * kernel_size, kernel_size=1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

    def forward(self, x, context):
        context_weights = self.context_conv(context)
        context_weights = context_weights.view(context_weights.size(0), -1, x.size(2), x.size(3))

        out = self.conv(x)
        return out + context_weights


