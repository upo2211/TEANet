import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
import torch.nn.functional as F

def worker_init_fn(worker_id):
    random.seed(2222 + worker_id)

def trainer_synapse(args, model, snapshot_path):#snapshot是保存路径
    from dataset_synapse import Synapse_dataset, RandomGenerator
    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(str(args))
    #定义初始学习率
    base_lr = args.base_lr
    num_classes = args.num_classes
    if args.n_gpu > 0:
        batch_size = args.batch_size * args.n_gpu
    else:
        batch_size = args.batch_size
    #加载数据集
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn,drop_last=True)

    #用GPU进行训练
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #=============================开始训练============================================
    model.train()
    #定义损失函数
    ce_loss = CrossEntropyLoss().to(device)
    dice_loss = DiceLoss(num_classes).to(device)
    #定义优化器
    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=1e-4)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = float('inf')
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch, edge_batch = sampled_batch['image'].repeat(1, 3, 1, 1), sampled_batch['label'], sampled_batch['edge']
            if torch.cuda.is_available():
                image_batch, label_batch, edge_batch = image_batch.cuda(), label_batch.cuda(), edge_batch.cuda()
            model = model.to(device)

            edge_output, seg_output = model(image_batch)
            loss_ce_seg = ce_loss(seg_output, label_batch[:].long())
            loss_ce_edge = ce_loss(edge_output, edge_batch[:].long())

            loss = 0.7*loss_ce_edge+0.3*loss_ce_seg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_seg', loss_ce_seg, iter_num)
            # writer.add_scalar('info/loss_edge', loss_ce_edge, iter_num)

            # normal loss print
            logging.info('iteration %d : loss : %f, loss_seg: %f' % (iter_num, loss.item(), loss_ce_seg.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(seg_output, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if loss<best_loss:
                best_loss=loss
                print("best_loss:", best_loss)
                save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), save_mode_path)

        save_interval = 50  # int(max_epoch/6)

        torch.cuda.empty_cache()

        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
