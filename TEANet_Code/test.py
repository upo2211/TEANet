import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_synapse import Synapse_dataset
from utils import test_single_volume
from MET_Net.baseline import METNet

from Comparision.UnetPP import *
from Comparision.TransUnet import *
# from Comparision.Polyp_PVT import PolypPVT
# from Comparision.SSFormer import mit_PLD_b4
from Comparision.deeplabv3_plus import DeepLab
from Comparision.UNet import UNet

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default=r'/data/source/record_lt/METNet/data_Synapse/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default=r'/data/source/record_lt/METNet/data_Synapse/list_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')

parser.add_argument('--is_savenii',type=bool, default=False, help='whether to save results during inference')
# parser.add_argument('--test_save_dir', type=str, default='D:\MET_Net\data_Synapse\list_Synapse', help='saving prediction as nii!')
args = parser.parse_args()


def inference(args, model,output_file, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    print("{} test iterations per epoch".format(len(testloader)))

    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        print('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        with open(output_file, 'a') as f:
            f.write('idx {} case {} mean_dice {:.6f} mean_hd95 {:.6f}\n'.format(
                i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))

    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        with open(output_file, 'a') as f:
            f.write('Mean class {} mean_dice {:.6f} mean_hd95 {:.6f}\n'.format(
                i, metric_list[i-1][0], metric_list[i-1][1]))

    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    with open(output_file, 'a') as f:
        f.write(
            'Testing performance in best val model: mean_dice {:.6f} mean_hd95 {:.6f}\n'.format(performance, mean_hd95))
    return "Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': r'/data/source/record_lt/METNet/data_Synapse/Synapse/test_vol_h5',
            'list_dir': r'/data/source/record_lt/METNet/data_Synapse/list_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']

    #======================定义模型以及加载预训练权重====================================
    # =====================METNet==============================
    net = METNet(n_channels=3, n_classes=9)
    #======================UNet=================================
    # net = UNet(n_channels=1, n_classes=args.num_classes)
    #======================UNetPP=================================
    # print("deep_supervision: False")
    # deep_supervision = False
    # net = UnetPlusPlus(num_classes=9, deep_supervision=deep_supervision)
    #======================TransUnet================================
    # config_vit = CONFIGS['R50-ViT-B_16']
    # config_vit.n_classes = 9
    # config_vit.n_skip = 3
    # img_size = 224
    # vit_patches_size = 16
    # config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    # net = VisionTransformer(config=config_vit, num_classes=9)
    # =======================Polyp_PVT=================================
    # net = PolypPVT().cuda()
    #=======================SSFormer==================================
    # net = mit_PLD_b4().cuda()
    #======================Deeplabv3==================================
    # net = DeepLab(num_classes=9).cuda()
    #==============================================================================

    model_path = r"/data/source/record_lt/METNet/model/Synapse224200_bs16_lr0.001_224_Ablation_baseline/epoch_199.pth"
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # 获取设备类型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 将模型移到设备
    net.to(device)

    args.exp = dataset_name + str(args.img_size)
    snapshot_path = "./output/{}".format(args.exp)
    snapshot_path = snapshot_path + str(args.max_epochs)#训练epoch
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)#batch_size
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)#初始学习率
    snapshot_path = snapshot_path + '_'+str(args.img_size)#图片大小
    snapshot_path = snapshot_path + '_Ablation_baseline'#模型名称
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # #创建保存nifty文件的文件夹
    # snapshot_nifty_path = "./nifty_save/{}".format(args.exp)
    # snapshot_nifty_path = snapshot_nifty_path + str(args.max_epochs)
    # snapshot_nifty_path = snapshot_nifty_path + '_bs'+str(args.batch_size)
    # snapshot_nifty_path = snapshot_nifty_path + '_lr'+str(args.base_lr)
    # snapshot_nifty_path = snapshot_nifty_path + '_'+str(args.img_size)
    # snapshot_nifty_path = snapshot_nifty_path + '_TransUnet_new'
    # if not os.path.exists(snapshot_nifty_path):
    #     os.makedirs(snapshot_nifty_path)

    # 创建一个 txt 文件用于输出
    output_file = os.path.join(snapshot_path, 'epoch_199.txt')
    with open(output_file, 'w') as f:
        # 写入初始输出内容
        f.write("Testing Started\n")
        f.write("Dataset: {}\n".format(dataset_name))
        f.write("Image Size: {}\n".format(args.img_size))
        f.write("Batch Size: {}\n".format(args.batch_size))
        f.write("Learning Rate: {}\n".format(args.base_lr))
        f.write("Max Epochs: {}\n".format(args.max_epochs))
        f.write("Number of Classes: {}\n".format(args.num_classes))

    test_save_path = None
    inference(args, net, output_file, test_save_path)


