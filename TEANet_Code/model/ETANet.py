import torchvision
import numpy as np
import cv2
from module import *


class ETANet(nn.Module):
    def __init__(self, n_channels, n_classes, deep_supervision=False, num_filters=64, dropout=False, rate=0.1,
                 bn=False):
        super(ETANet, self).__init__()
        self.deep_supervision = deep_supervision
        self.res_model = torchvision.models.resnet18(pretrained=True)
        self.res_features = torch.nn.Sequential(*list(self.res_model.children())[:-2])
        self.inc = self.res_features[:3]
        self.down1 = self.res_features[3:5]
        self.down2 = self.res_features[5:6]
        self.maxpool1 = nn.MaxPool2d(2)
        self.down3 = DBTB(c1=128, c2=256, tcr=0.25)
        self.maxpool2 = nn.MaxPool2d(2)
        self.down4 = DBTB(c1=256, c2=512, tcr=0.5)
        self.sigmoid = nn.Sigmoid()

        # boundray stream
        self.edgeconv1 = DoubleConv(512, 256)
        self.edgeconv2 = DoubleConv(512, 128)
        self.edgeconv3 = DoubleConv(256, 64)
        self.edgeconv4 = DoubleConv(128, 64)
        self.edgeconv5 = DoubleConv(128, 64)

        # self.edgeup = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.edgeup1 = Up(256, bilinear=True)
        self.edgeup2 = Up(128, bilinear=True)
        self.edgeup3 = Up(64, bilinear=True)
        self.edgeup4 = Up(64, bilinear=True)
        self.edgeup5 = Up(64, bilinear=True)

        self.ee1 = EdgeEnhancer(256)
        self.ee2 = EdgeEnhancer(128)
        self.ee3 = EdgeEnhancer(64)
        self.ee4 = EdgeEnhancer(64)
        self.ee5 = EdgeEnhancer(64)

        self.final_conv = nn.Conv2d(64, 2, kernel_size=3, padding=1)

        self.cw = nn.Conv2d(3, 1, kernel_size=1, padding=0, bias=False)

        self.se = SE_Block(64)

        # segmentation stream
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1 = Up(256, bilinear=True)
        self.up2 = Up(128, bilinear=True)
        self.up3 = Up(64, bilinear=True)
        self.up4 = Up(64, bilinear=True)
        self.up5 = Up(192, bilinear=True)

        self.conv1 = DoubleConv(512, 256)
        self.conv2 = DoubleConv(768, 128)
        self.conv3 = DoubleConv(384, 64)
        self.conv4 = DoubleConv(192, 64)

        self.c1 = nn.Sequential(nn.Conv2d(1, 256, kernel_size=3, padding=1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(16))
        self.c2 = nn.Sequential(nn.Conv2d(1, 128, kernel_size=3, padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(8))
        self.c3 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(4))
        self.c4 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2))

        self.outconv = nn.Conv2d(192, n_classes, kernel_size=1)

        self.weight_conv_1 = nn.Conv2d(256, 3, kernel_size=1)
        self.weight_conv_2 = nn.Conv2d(128, 3, kernel_size=1)
        self.weight_conv_3 = nn.Conv2d(64, 3, kernel_size=1)
        self.weight_conv_4 = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        x_size = x.size()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3_ = self.maxpool1(x3)
        x4 = self.down3(x3_)
        x4_ = self.maxpool2(x4)
        x5 = self.down4(x4_)

        # ========================edge detection stream=============================
        xe5 = self.ee1(self.edgeup1(self.edgeconv1(x5), x4))
        x4p = torch.cat([xe5, x4], 1)
        xe4 = self.ee2(self.edgeup2(self.edgeconv2(x4p), x3))
        x3p = torch.cat([xe4, x3], 1)
        xe3 = self.ee3(self.edgeup3(self.edgeconv3(x3p), x2))
        x2p = torch.cat([xe3, x2], 1)
        xe2 = self.ee4(self.edgeup4(self.edgeconv4(x2p), x1))
        x1p = torch.cat([xe2, x1], 1)
        xe1 = self.ee5(self.edgeup5(self.edgeconv5(x1p), x))
        xe_se = self.se(xe1)  # 过一个通道注意力机制
        xe_out = self.final_conv(xe_se)
        x_edge = self.sigmoid(xe_out)

        ### Canny Edge
        im_arr = np.mean(x.cpu().numpy(), axis=1).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        if torch.cuda.is_available():
            canny = torch.from_numpy(canny).cuda().float()  # 注意.cuda()
        else:
            canny = torch.from_numpy(canny).float()
        ### End Canny Edge

        cat = torch.cat([x_edge, canny], dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)
        # ========================segmentation stream=============================
        edge1 = Correction(self.c1(acts), x4)
        edge2 = Correction(self.c2(acts), x3)
        edge3 = Correction(self.c3(acts), x2)
        edge4 = Correction(self.c4(acts), x1)

        # 1
        x5_ = self.up1(self.conv1(x5),x4)
        fusion_input_1 = x4+x5_+edge1
        weights_1 = torch.sigmoid(self.weight_conv_1(fusion_input_1))
        x4w, x5_w, edge1w = weights_1.split(1, dim=1)
        x4 = x4w*x4
        x5_ = x5_w*x5_
        edge1 = edge1w*edge1
        x4i = torch.cat([x4, x5_, edge1], dim=1)

        # 2
        x4_ = self.up2(self.conv2(x4i), x3)
        fusion_input_2 = x3+x4_+edge2
        weights_2 = torch.sigmoid(self.weight_conv_2(fusion_input_2))
        x3w, x4_w, edge2w = weights_2.split(1, dim=1)
        x3 = x3w*x3
        x4_ = x4_w*x4_
        edge2 = edge2w*edge2
        x3i = torch.cat([x3, x4_, edge2], dim=1)

        # 3
        x3_ = self.up3(self.conv3(x3i), x2)

        fusion_input_3 = x2+x3_+edge3
        weights_3 = torch.sigmoid(self.weight_conv_3(fusion_input_3))
        x2w, x3_w, edge3w = weights_3.split(1, dim=1)
        x2 = x2w*x2
        x3_ = x3_w*x3_
        edge3 = edge3w*edge3
        x2i = torch.cat([x2, x3_, edge3], dim=1)

        # 4
        x2_ = self.up4(self.conv4(x2i), x1)
        fusion_input4 = x1+x2_+edge4
        weights_4 = torch.sigmoid(self.weight_conv_4(fusion_input4))
        x1w, x2_w, edge4w = weights_4.split(1, dim=1)
        x1 = x1w*x1
        x2_ = x2_w*x2_
        edge4 = edge4w*edge4
        x1i = torch.cat([x1, x2_, edge4], dim=1)
        x1iu = self.up5(x1i, x)
        xs_out = self.outconv(x1iu)

        return xe_out, xs_out



if __name__ == '__main__':
    ras = ETANet(n_channels=1, n_classes=9)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ras.to(device)
    input_tensor = torch.randn(1, 3, 129, 320)
    xe,xs = ras(input_tensor)
    print(xs.shape)