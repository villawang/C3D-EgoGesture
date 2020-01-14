import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm, trange
import shutil
from random import randint
import argparse
import glob
import pdb
import random
import time
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy


import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader,RandomSampler
from torchsummary import summary


from model import C3D 
from opts import parse_opts
import utils
from transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from R3D_dataset import dataset_task_R3D, dataset_R3D
import dataset_unequal

import warnings
warnings.filterwarnings("ignore")


# root_dir = '/home/zhengwei/dataset/egogesture'
root_dir = '/home/zhengwei/dataset/egogesture/FramesInMedia'
save_dir = 'output'
model_dir = './model'

device = 'cuda:2'
torch.cuda.set_device(2)

args = parse_opts()

def forward(net, data):
    inputs, masks, labels = data
    inputs, labels = inputs.to(device, non_blocking=True).float(
    ), labels.to(device, non_blocking=True).long()
    outputs, logits = net(inputs)
    return outputs, logits, labels

def fine_tune(load_dir, save_dir, filename, dataloader_train, dataloader_train_sampler, 
              dataloader_val):
    # load pretrained model
    pretrained_weights = torch.load(os.path.join(load_dir, filename), map_location='cpu')
    feature_state_dict = {key: value for key, value in pretrained_weights.items() 
                if key not in ['fc8.weight', 'fc8.bias']}
    net = C3D.C3D()
    state_dict = deepcopy(net.state_dict())
    state_dict.update(feature_state_dict)
    net.load_state_dict(state_dict)
    net.to(device)
    net.train()


    # pdb.set_trace()

    summary(net, (3,args.n_frames_per_clip,112,112))

    # pdb.set_trace()

    num_epochs = 100
    step = 0
    # determine optimizer
    criterion = nn.CrossEntropyLoss()
    fc_lr_layers = list(map(id, net.fc8.parameters()))
    pretrained_lr_layers = [p for p in net.parameters() 
                            if id(p) not in fc_lr_layers and p.requires_grad==True]
    # pretrained_lr_layers = filter(lambda p: 
    #                               id(p) not in fc_lr_layers, model.parameters())
    optimizer = torch.optim.SGD([
        {"params": net.fc8.parameters()},
        {"params": pretrained_lr_layers, "lr": 1e-4, 'weight_decay':5e-3}
    ], lr=1e-3, momentum=0.9, weight_decay=1e-2)

    # optimizer = torch.optim.SGD(net.fc8.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)

    # optimizer = torch.optim.Adam([
    #     {"params": model.module.fc.parameters()},
    #     {"params": pretrained_lr_layers, "lr": 1e-4}
    # ], lr=1e-3)


    # optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9, weight_decay=5e-4)

    train_logger = utils.Logger(os.path.join('output', 'C3D-fine-tune-all.log'),
                                ['step', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 
                                'lr_feature', 'lr_fc'])
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    train_loss = utils.AverageMeter()
    train_acc = utils.AverageMeter()
    val_loss = utils.AverageMeter()
    val_acc = utils.AverageMeter()
    batch_cache = 1
    for epoch in trange(num_epochs):  # loop over the dataset multiple times
        train_loss.reset()
        train_acc.reset()
        for data in dataloader_train:
            outputs, logits, labels = forward(net, data)
            # print(batch_cache)
            # pdb.set_trace()
            # if batch_cache == 8: # accumulate number of batches because gpu memory limits the batch size
            #     optimizer.step()
            #     optimizer.zero_grad()
            loss_ = criterion(logits, labels)
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()
            train_loss.update(loss_.item())
            train_acc.update(utils.calculate_accuracy(outputs, labels))
            # if batch_cache == 8:
            #     batch_cache = 0
            if step % 100 == 0:
                val_loss.reset()
                val_acc.reset()
                net.eval()
                # for data_sampler in dataloader_train_sampler:
                #     outputs_sampler, logits_sampler, labels_sampler = forward(net, data_sampler)
                #     loss_sampler_ = criterion(logits_sampler, labels_sampler)
                #     train_loss.update(loss_sampler_.item())
                #     train_acc.update(utils.calculate_accuracy(outputs_sampler, labels_sampler))
                for data_val in dataloader_val:
                    outputs_val, logits_val, labels_val = forward(net, data_val)
                    val_loss_ = criterion(logits_val, labels_val)
                    val_loss.update(val_loss_.item())
                    val_acc.update(utils.calculate_accuracy(outputs_val, labels_val))
                net.train()
                print('epoch{}/{} train_acc:{:.3f} train_loss:{:.3f} val_acc:{:.3f} val_loss:{:.3f}'.format(
                    epoch + 1, num_epochs,
                    train_acc.val, train_loss.val,
                    val_acc.avg, val_loss.avg
                    ))
                train_logger.log({
                    'step': step,
                    'train_loss': train_loss.val,
                    'train_acc': train_acc.val,
                    'val_loss': val_loss.avg,
                    'val_acc': val_acc.avg,
                    # 'lr_feature': optimizer.param_groups[1]['lr'],
                    'lr_feature': 0,
                    'lr_fc': optimizer.param_groups[0]['lr']
                })
            if step % 200 == 0:
                utils.save_checkpoint(net, optimizer, step, save_dir,
                                'c3d.pth')
            step += 1                     
            # batch_cache += 1           
        scheduler.step()

def main(args):
    # keep shuffling be constant every time
    seed = 1
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


    # # norm_method = Normalize(args.mean, args.std)
    # norm_method = Normalize([0, 0, 0], [1, 1, 1])
    # # scales = [0.5, 0.6, 0.7, 0.8, 0.9]
    # trans_train = Compose([
    #             # Scale(args.img_size),
    #             # MultiScaleRandomCrop(scales, args.img_size),
    #             # CenterCrop(args.img_size),
    #             RandomHorizontalFlip(),
    #             ToTensor(1), norm_method
    #             ])
    # trans_test = Compose([
    #             # Scale(args.img_size),
    #             # CenterCrop(args.img_size),
    #             ToTensor(1), norm_method
    #             ])



    # mask_trans = transforms.Compose([
    #             # transforms.Resize((126,224)),
    #             # transforms.CenterCrop((126,224)),
    #             transforms.ToTensor()
    #             ])

    mask_trans = transforms.Compose([
                    # transforms.Resize((126,224)),
                    # transforms.CenterCrop((126,224)),
                    transforms.ToTensor()
                    ])
    trans_train = transforms.Compose([
                    # transforms.Resize((126,224)),
                    # transforms.RandomResizedCrop(224, 126),
                    # transforms.CenterCrop((126,224)),
                    #     transforms.RandomSizedCrop(255),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    # [0.229, 0.224, 0.225]
                ])
    trans_test = transforms.Compose([
                    # transforms.Resize((126,224)),
                    # transforms.CenterCrop((126,224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])



    print('Loading phase2 training data.....')
    dataset_train = dataset_unequal.dataset_all(root_dir, 'train',
                                        n_frames_per_clip=args.n_frames_per_clip,
                                        UnequalSequence = True,
                                        img_size=(args.w, args.h), stride=2,
                                        reverse=False, transform=trans_train,
                                        mask_trans=mask_trans)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                    shuffle=True, 
                                    num_workers=args.num_workers, pin_memory=True)
    sampler = RandomSampler(dataset_train, replacement=True, num_samples=1024)               
    dataloader_train_sampler = DataLoader(dataset_train, batch_size=4,
                                    sampler=sampler, 
                                    num_workers=args.num_workers, pin_memory=True)

    print('\n')
    print('Loading phase2 validating data.....')
    dataset_val = dataset_unequal.dataset_all(root_dir, 'val', 
                                        n_frames_per_clip = args.n_frames_per_clip, 
                                        img_size=(args.w, args.h), stride=args.n_frames_per_clip, 
                                        UnequalSequence = True,
                                        reverse=False, transform=trans_test, 
                                        mask_trans = mask_trans)
    sampler = RandomSampler(dataset_val, replacement=True, num_samples=1024)
    dataloader_val = DataLoader(dataset_val, batch_size=12, 
                                sampler=sampler,num_workers=args.num_workers,pin_memory=True)

    fine_tune(model_dir, save_dir, 'c3d.pickle',
              dataloader_train, dataloader_train_sampler,
              dataloader_val)

if __name__ == '__main__':
    main(args)