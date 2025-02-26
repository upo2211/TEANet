import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from model.ETANet import ETANet
from trainer import trainer_synapse


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default=r'D:\TEANet_Code\data_Synapse\Synapse\train_npz', help='root dir for data')#your training data path!
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default=r'D:\TEANet_Code\data_Synapse\list_Synapse', help='list dir')#your list data path!
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=0, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')
args = parser.parse_args()


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
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path':args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True

    args.exp = dataset_name + str(args.img_size)
    snapshot_path = "./model/{}".format(args.exp)
    snapshot_path = snapshot_path + str(args.max_epochs)#训练epoch
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)#batch_size
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)#初始学习率
    snapshot_path = snapshot_path + '_'+str(args.img_size)#图片大小
    snapshot_path = snapshot_path + '_TEANet'#模型名称
    model_dir = os.path.join(os.getcwd(), 'model')
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    net = ETANet(n_channels=3, n_classes=9)

    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](args, net, snapshot_path)