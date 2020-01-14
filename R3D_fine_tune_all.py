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


from model import resnet 
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
model_dir = '/home/zhengwei/dataset/UCF-101/pretrained_model/C3D/R3D-pretrained'

device = 'cuda:1'
torch.cuda.set_device(1)

args = parse_opts()





def model_test(save_dir, filename, dataloader):
    criterion = nn.CrossEntropyLoss()
    checkpoint = utils.load_checkpoint(save_dir, filename)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    valid_acces = []
    valid_losses = []
    for data in tqdm(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device, non_blocking=True).float(
        ), labels.to(device, non_blocking=True).long()
        outputs = model(inputs)
        _, pred = outputs.max(1)
        loss_ = criterion(outputs, labels)
        _, pred = outputs.max(1)
        num_correct = (pred == labels).sum()
        acc = num_correct / torch.Tensor([inputs.shape[0]]).to(device)
        valid_acces.append(acc.item())
        valid_losses.append(loss_.item())
    print(' val_acc:{:.3f} val_loss:{:.4f}'.format(
            np.mean(valid_acces),
            np.mean(valid_losses)))





def fine_tune(load_dir, save_dir, filename, dataloader_train, dataloader_val, ContinuousTrain=False):
    '''
    load pretrained model
    resnet-18-kinetics.pth: --model resnet --model_depth 18 --resnet_shortcut A
    resnet-34-kinetics.pth: --model resnet --model_depth 34 --resnet_shortcut A
    resnet-34-kinetics-cpu.pth: CPU ver. of resnet-34-kinetics.pth
    resnet-50-kinetics.pth: --model resnet --model_depth 50 --resnet_shortcut B
    resnet-101-kinetics.pth: --model resnet --model_depth 101 --resnet_shortcut B
    resnet-152-kinetics.pth: --model resnet --model_depth 152 --resnet_shortcut B
    resnet-200-kinetics.pth: --model resnet --model_depth 200 --resnet_shortcut B
    preresnet-200-kinetics.pth: --model preresnet --model_depth 200 --resnet_shortcut B
    wideresnet-50-kinetics.pth: --model wideresnet --model_depth 50 --resnet_shortcut B --wide_resnet_k 2
    resnext-101-kinetics.pth: --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32
    densenet-121-kinetics.pth: --model densenet --model_depth 121
    densenet-201-kinetics.pth: --model densenet --model_depth 201
    '''
    num_epochs = 100
    step = 0
    if not ContinuousTrain:
        model = resnet.resnet152(sample_size=112,
                                sample_duration=args.n_frames_per_clip, shortcut_type='B',
                                num_classes=83)
        checkpoint = utils.load_checkpoint(load_dir, filename)
        model=nn.DataParallel(model, device_ids=[1])
        state_dict = deepcopy(model.state_dict())
        feature_state_dict = {key: value for key, value in checkpoint['state_dict'].items() 
                    if key not in ['module.fc.weight', 'module.fc.bias']}
        state_dict.update(feature_state_dict)
        model.load_state_dict(state_dict)


        # set fine tune parameters: Conv5_x and fc layer from original paper
        for param in model.module.parameters():
            param.requires_grad = False
        for named_child in model.module.named_children():
            if named_child[0] == 'fc' or named_child[0] == 'layer4':
            # if named_child[0] == 'fc':
                for param in named_child[1].parameters():
                    param.requires_grad = True
        # pdb.set_trace()
    else:
        print('Recover model from {}/resnet18.pth......'.format(save_dir))
        checkpoint = utils.load_checkpoint(save_dir, 'resnet18.pth')
        model = checkpoint['model']
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        step = checkpoint['step']


    model.to(device)
    summary(model, (3,args.n_frames_per_clip,112,112))

    # pdb.set_trace()

    model.train()

    # determine optimizer
    criterion = nn.CrossEntropyLoss()
    fc_lr_layers = list(map(id, model.module.fc.parameters()))
    pretrained_lr_layers = [p for p in model.parameters() 
                            if id(p) not in fc_lr_layers and p.requires_grad==True]
    # pretrained_lr_layers = filter(lambda p: 
    #                               id(p) not in fc_lr_layers, model.parameters())
    optimizer = torch.optim.SGD([
        {"params": model.module.fc.parameters()},
        {"params": pretrained_lr_layers, "lr": 1e-4, 'weight_decay':1e-3}
    ], lr=1e-3, momentum=0.9, weight_decay=1e-2)

    train_logger = utils.Logger(os.path.join('output', 'R3D-fine-tune-all.log'),
                                ['step', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 
                                'lr_feature', 'lr_fc'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    train_loss = utils.AverageMeter()
    train_acc = utils.AverageMeter()
    val_loss = utils.AverageMeter()
    val_acc = utils.AverageMeter()
    for epoch in trange(num_epochs):  # loop over the dataset multiple times
        train_loss.reset()
        train_acc.reset()
        for data in dataloader_train:
            inputs, masks, labels = data
            inputs, labels = inputs.to(device, non_blocking=True).float(
            ), labels.to(device, non_blocking=True).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_ = criterion(outputs, labels)
            loss_.backward()
            optimizer.step()
            train_loss.update(loss_.item())
            train_acc.update(utils.calculate_accuracy(outputs, labels))
            if step % 50 == 0:
                val_loss.reset()
                val_acc.reset()
                model.eval()
                for data_val in dataloader_val:
                    inputs_val, masks_val, labels_val = data_val
                    inputs_val, labels_val = inputs_val.to(device, non_blocking=True).float(), \
                                            labels_val.to(device, non_blocking=True).long()
                    outputs_val = model(inputs_val)
                    val_loss_ = criterion(outputs_val, labels_val)
                    val_loss.update(val_loss_.item())
                    val_acc.update(utils.calculate_accuracy(outputs_val, labels_val))
                model.train()
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
                utils.save_checkpoint(model, optimizer, step, save_dir,
                                'resnet18.pth')
            step += 1            
        scheduler.step()


    

def main(args):
    # keep shuffling be constant every time
    seed = 1
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


    # norm_method = Normalize(args.mean, args.std)
    norm_method = Normalize([0, 0, 0], [1, 1, 1])
    # scales = [0.5, 0.6, 0.7, 0.8, 0.9]
    trans_train = Compose([
                # Scale(args.img_size),
                # MultiScaleRandomCrop(scales, args.img_size),
                # CenterCrop(args.img_size),
                RandomHorizontalFlip(),
                ToTensor(1), norm_method
                ])
    trans_test = Compose([
                # Scale(args.img_size),
                # CenterCrop(args.img_size),
                ToTensor(1), norm_method
                ])

    if args.is_train:
        # dataset_train = dataset_R3D(
        #     root_dir, 'train', args.data_type, args.n_frames_per_clip, 
        #     img_size=args.img_size, stride=1, overlap=True, reverse=True, transform=trans_train)
        # dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, 
        #                             shuffle=True,num_workers=12,pin_memory=True)

        # dataset_val = dataset_R3D(
        #     root_dir, 'val', args.data_type, args.n_frames_per_clip, 
        #     img_size=args.img_size, stride=1, overlap=False, transform=trans_test)
        # sampler = RandomSampler(dataset_val, replacement=True, num_samples=args.batch_size*5)
        # dataloader_val = DataLoader(dataset_val, batch_size=16, 
        #                             num_workers=8,sampler=sampler,
        #                             pin_memory=True)
        # dataloader for phase 2
        mask_trans = transforms.Compose([
                    # transforms.Resize((126,224)),
                    # transforms.CenterCrop((126,224)),
                    transforms.ToTensor()
                    ])
        print('Loading phase2 training data.....')
        dataset_train = dataset_unequal.dataset_all(root_dir, 'train',
                                            n_frames_per_clip=args.n_frames_per_clip,
                                            UnequalSequence = True,
                                            img_size=(args.w, args.h), stride=2,
                                            reverse=False, transform=trans_train,
                                            mask_trans=mask_trans)
        dataloader_train = DataLoader(dataset_train, batch_size=128,
                                        shuffle=True, 
                                        num_workers=args.num_workers, pin_memory=True)

        print('\n')
        print('Loading phase2 validating data.....')
        dataset_val = dataset_unequal.dataset_all(root_dir, 'val', 
                                            n_frames_per_clip = args.n_frames_per_clip, 
                                            img_size=(args.w, args.h), stride=2, 
                                            UnequalSequence = True,
                                            reverse=False, transform=trans_test, 
                                            mask_trans = mask_trans)
        sampler = RandomSampler(dataset_val, replacement=True, num_samples=1024)
        dataloader_val = DataLoader(dataset_val, batch_size=64, 
                                    sampler=sampler,num_workers=args.num_workers,pin_memory=True)
    else:
        dataset_test = dataset_R3D(
            root_dir, 'test', args.data_type, args.n_frames_per_clip, 
            img_size=args.img_size, stride=args.n_frames_per_clip, overlap=False, transform=trans_test)
        dataloader_test = DataLoader(dataset_test, batch_size=128, 
                                    shuffle=True,num_workers=8,pin_memory=True)
    if args.is_train:
        fine_tune(model_dir, save_dir, 'resnet-152-kinetics.pth',
                  dataloader_train, dataloader_val, ContinuousTrain=False)
        pdb.set_trace()
    else:
        model_test(
             'output', 'checkpoint-3.pth', dataloader_test)
        # model_test_pretrained(
        #     model_dir, 'resnet-34-kinetics-ucf101_split1.pth', dataloader_test)




if __name__ == '__main__':
    main(args)




# def model_test_pretrained(save_dir, filename, dataloader):
#     model = resnet.resnet34(sample_size=args.img_size,
#                             sample_duration=args.n_frames_per_clip,
#                             shortcut_type='A', num_classes=83)
#     checkpoint = utils.load_checkpoint(save_dir, filename)
#     # model = nn.DataParallel(model, device_ids=[0])
#     state_dict = model.state_dict()
#     state_dict.update(checkpoint['state_dict'])
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()
#     valid_acces = []
#     valid_losses = []
#     criterion = nn.CrossEntropyLoss()
#     for data in tqdm(dataloader):
#         inputs, labels = data
#         # pdb.set_trace()

#         inputs, labels = inputs.to(device, non_blocking=True).float(
#         ), labels.to(device, non_blocking=True).long()
#         outputs = model(inputs)
#         _, pred = outputs.max(1)
#         loss_ = criterion(outputs, labels)
#         _, pred = outputs.max(1)
#         num_correct = (pred == labels).sum()
#         acc = num_correct / torch.Tensor([inputs.shape[0]]).to(device)
#         valid_acces.append(acc.item())
#         valid_losses.append(loss_.item())
#     print(' val_acc:{:.3f} val_loss:{:.4f}'.format(
#         np.mean(valid_acces),
#         np.mean(valid_losses)))