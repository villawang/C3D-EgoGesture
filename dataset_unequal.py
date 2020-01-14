import os 
import sys
sys.path.append(os.getcwd()[0:-7])
sys.path.append(os.path.join(os.getcwd()[0:-7], 'utils'))
import pickle
import numpy as np
import pandas as pd
import random
import torch
import pdb
from torch.utils.data import Dataset, DataLoader,RandomSampler
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import random
import skimage.util as ski_util
from sklearn.utils import shuffle

# self-defined modules
import opts



# args = opts.parse_opts()
root_path = '/home/zhengwei/dataset/egogesture/FramesInMeadia'

# # load annotation files task by task
# def load_frame_task(csv_path, mode, n_frames_per_clip, 
#                     task, stride=1, reverse=False):
#     # mode: train, val, test
#     csv_file = os.path.join(root_path, '{}.csv'.format(mode))
#     annot_df = pd.read_csv(csv_file)
#     # get task index in dataframe
#     task_ind = []
#     for i in range(annot_df.shape[0]):
#         if eval(annot_df['rgb'][i])[0].split('/')[-4] == task:
#             task_ind.append(i)
#     annot_df_task = annot_df.iloc[task_ind]
#     rgb_samples = []
#     depth_samples = []
#     labels = []
#     for frame_i in range(annot_df_task.shape[0]):
#         rgb_list = eval(annot_df_task['rgb'].iloc[frame_i]) # convert string in dataframe to list
#         depth_list = eval(annot_df_task['depth'].iloc[frame_i])
#         if len(rgb_list) >= n_frames_per_clip:
#             # define how many samples for a video clip
#             clip_final_ind = int(
#                 len(rgb_list) - n_frames_per_clip + 1 - stride)
#             clip_i = 0
#             while clip_i <= clip_final_ind:
#                 rgb_samples.append(
#                     rgb_list[clip_i:(clip_i + n_frames_per_clip)])
#                 depth_samples.append(
#                     depth_list[clip_i:(clip_i + n_frames_per_clip)])    
#                 labels.append(annot_df['label'].iloc[frame_i])
#                 # data augmentation by reversing the sequence of the video
#                 if reverse:
#                     rgb_samples.append(
#                         rgb_list[::-1][clip_i:(clip_i + n_frames_per_clip)])
#                     depth_samples.append(
#                         depth_list[::-1][clip_i:(clip_i + n_frames_per_clip)])
#                     labels.append(annot_df['label'].iloc[frame_i])
#                 clip_i += stride
#     return rgb_samples, depth_samples, labels



# # load frame clip for one task each time
# class dataset_task(Dataset):
#     def __init__(self, root_path, mode, n_frames_per_clip, task, img_size, 
#                  stride=1, reverse=False, transform=None, mask_trans=None):
#         self.root_path = root_path
#         self.rgb_samples, self.depth_samples, self.labels = load_frame_task(root_path, mode, 
#                                                                             n_frames_per_clip,
#                                                                             task, 
#                                                                             stride, reverse)
#         self.w = img_size[0]
#         self.h = img_size[1]
#         self.sample_num = len(self.rgb_samples)
#         self.n_frames_per_clip = n_frames_per_clip
#         self.transform = transform
#         self.mask_transform = mask_trans
#         print('{} {} samples have been loaded'.format(task, self.sample_num))

#     def __getitem__(self, idx):
#         rgb_name = self.rgb_samples[idx]
#         depth_name = self.depth_samples[idx]
#         label = self.labels[idx]
#         rgb = torch.zeros([3, self.n_frames_per_clip, self.h, self.w])
#         masks = torch.empty([self.n_frames_per_clip, self.h, self.w], dtype=torch.long)

#         for frame_name_i in range(len(rgb_name)):
#             rgb_cache = Image.open(rgb_name[frame_name_i]).convert("RGB").resize((self.w, self.h), Image.BILINEAR)  # C * W * H
#             rgb_cache = self.transform(rgb_cache)
#             rgb[:, frame_name_i, :, :] = rgb_cache
#             mask = Image.open(depth_name[frame_name_i]).convert("L").resize((self.w, self.h), Image.BILINEAR)  # C * W * H
#             threshold = 10
#             mask = mask.point(lambda p: p > threshold and 255)
#             mask = self.mask_transform(mask)
#             mask = torch.squeeze(mask)
#             masks[frame_name_i, :, :] = mask
#         if len(rgb_name) == 1: # for images
#             return torch.squeeze(rgb), mask, int(int(label) - 1)
#         else: # for videos
#             return rgb, masks, int(int(label) - 1)
#         # return rgb, masks, (torch.tensor(label)-1).long()

#     def __len__(self):
#         return int(self.sample_num)

# This script downsamples frames larger than 40 to 40 (follow the original paper)
def DownSample(annot_df, sample_num):
    downsampled_annot_dict = {k: [] for k in ['rgb', 'depth', 'label']}
    for video_i in range(annot_df.shape[0]):
        if len(annot_df['rgb'].iloc[video_i]) > sample_num:
            LastPossibleStart = len(annot_df['rgb'].iloc[video_i]) - sample_num
            start = random.randint(0,LastPossibleStart)
            downsampled_annot_dict['rgb'].append(annot_df['rgb'].iloc[video_i][start:start+sample_num])
            downsampled_annot_dict['depth'].append(annot_df['depth'].iloc[video_i][start:start+sample_num])
            downsampled_annot_dict['label'].append(annot_df['label'].iloc[video_i])
        else:
            downsampled_annot_dict['rgb'].append(annot_df['rgb'].iloc[video_i])
            downsampled_annot_dict['depth'].append(annot_df['depth'].iloc[video_i])
            downsampled_annot_dict['label'].append(annot_df['label'].iloc[video_i])
        assert len(downsampled_annot_dict['rgb'][video_i]) <= sample_num
    return pd.DataFrame(downsampled_annot_dict)


def load_frame_all(csv_path, mode, n_frames_per_clip, UnequalSequence = False, 
                   reverse=False, stride=1):
    # UnequalSequence is to control whether the sequence shorter than n_frames_per_clip
    # will be selected.
    # mode: train, val, test
    csv_file = os.path.join(root_path, '{}.pkl'.format(mode))
    raw_annot_df = pd.read_pickle(csv_file)
    annot_df = DownSample(raw_annot_df, 40)
    # pdb.set_trace()
    rgb_samples = []
    depth_samples = []
    labels = []
    for frame_i in range(annot_df.shape[0]):
        rgb_list = annot_df['rgb'].iloc[frame_i] # convert string in dataframe to list
        depth_list = annot_df['depth'].iloc[frame_i] # convert string in dataframe to list
        if len(rgb_list) >= n_frames_per_clip:
            for clip_i in range(len(rgb_list)):
                start = clip_i*stride
                end = start + n_frames_per_clip
                if end > len(rgb_list):
                    # attach rest of samples
                    if len(rgb_list)-start >= 8: # set the threshold of rested samples
                        rgb_samples.append(rgb_list[start:])
                        depth_samples.append(depth_list[start:])
                        labels.append(annot_df['label'].iloc[frame_i])
                    break
                rgb_samples.append(rgb_list[start:end])
                depth_samples.append(depth_list[start:end])
                labels.append(annot_df['label'].iloc[frame_i])
                if reverse:
                    rgb_samples.append(
                        rgb_list[::-1][clip_i:(clip_i + n_frames_per_clip)])
                    depth_samples.append(
                        depth_list[::-1][clip_i:(clip_i + n_frames_per_clip)])
                    labels.append(annot_df['label'].iloc[frame_i])
            # pdb.set_trace()
        elif len(rgb_list) < n_frames_per_clip: 
        # elif len(rgb_list) < n_frames_per_clip: 
            if UnequalSequence: # load frames shorter than n_frames_per_clip
                rgb_samples.append(rgb_list)
                depth_samples.append(depth_list)
                labels.append(annot_df['label'].iloc[frame_i])
                if reverse:
                    rgb_samples.append(rgb_list[::-1])
                    depth_samples.append(depth_list[::-1])
                    labels.append(annot_df['label'].iloc[frame_i])
    return rgb_samples, depth_samples, labels

# def load_frame_all(csv_path, mode, n_frames_per_clip, UnequalSequence = False, 
#                    reverse=False, stride=1):
#     # UnequalSequence is to control whether the sequence shorter than n_frames_per_clip
#     # will be selected.
#     # mode: train, val, test
#     # csv_file = os.path.join(root_path, '{}.csv'.format(mode))
#     # annot_df = pd.read_csv(csv_file)

#     pkl_file = os.path.join(root_path, '{}.pkl'.format(mode))
#     annot_df = pd.read_pickle(pkl_file)

#     # pkl_file_wrapper = os.path.join(root_path, '{}.pkl'.format('all_id_wrapper'))
#     # annot_df_wrapper = pd.read_pickle(pkl_file_wrapper)

#     # pkl_file = os.path.join(root_path, '{}.pkl'.format('all_id'))
#     # annot_df = pd.read_pickle(pkl_file)



#     rgb_samples = []
#     depth_samples = []
#     labels = []
#     annot_dict = {k: [] for k in ['rgb', 'depth', 'label']}
#     for frame_i in range(annot_df.shape[0]):
#         rgb_list = annot_df['rgb'].iloc[frame_i] # convert string in dataframe to list
#         depth_list = annot_df['depth'].iloc[frame_i] # convert string in dataframe to list
#         for img_i in range(len(rgb_list)):
#             rgb_samples.append(rgb_list[img_i])
#             depth_samples.append(depth_list[img_i])
#             labels.append(annot_df['label'].iloc[frame_i])

#             # annot_dict['rgb'].append(rgb_list[img_i])
#             # annot_dict['depth'].append(depth_list[img_i])
#             # annot_dict['label'].append(annot_df_wrapper['label'].iloc[frame_i])
#     # from pandas.util.testing import assert_frame_equal
#     # pdb.set_trace()
#     # assert_frame_equal(pd.DataFrame(annot_dict), annot_df)
#     return rgb_samples, depth_samples, labels
#     # return annot_df['rgb'].tolist(), annot_df['depth'].tolist(), annot_df['label'].tolist()

    

# load frame clip for all tasks
class dataset_all(Dataset):
    def __init__(self, root_path, mode, n_frames_per_clip, img_size, 
                stride, UnequalSequence = False,
                reverse=False, transform=None, mask_trans=None):
        self.root_path = root_path
        self.rgb_samples, self.depth_samples, self.labels = load_frame_all(root_path, mode, 
                                                            n_frames_per_clip, 
                                                            UnequalSequence,
                                                            reverse, stride)
        # random_seed = 10
        # random.seed(random_seed)
        # indexes = list(range(len(rgb_samples)))
        # random.shuffle(indexes)
        # self.rgb_samples, self.depth_samples, self.labels = rgb_samples[indexes], depth_samples[indexes], labels[indexes]
        # self.rgb_samples, self.depth_samples, self.labels = shuffle(rgb_samples, depth_samples, labels, random_state=10)
        self.w = img_size[0]
        self.h = img_size[1]
        self.sample_num = len(self.rgb_samples)
        self.n_frames_per_clip = n_frames_per_clip
        self.transform = transform
        self.mask_transform = mask_trans
        print('{} samples have been loaded'.format(self.sample_num))

    def __getitem__(self, idx):
        rgb_name = self.rgb_samples[idx]
        depth_name = self.depth_samples[idx]
        label = self.labels[idx]
        rgb = torch.zeros([3, self.n_frames_per_clip, self.h, self.w])
        masks = torch.zeros([self.n_frames_per_clip, self.h, self.w])
        assert len(rgb_name) <= self.n_frames_per_clip
        # repeat frame sequence less than the n_frames_per_clip
        if len(rgb_name) < self.n_frames_per_clip: 
            RepeatTimes = int(self.n_frames_per_clip/len(rgb_name))+1
            rgb_name = rgb_name*RepeatTimes
            depth_name = depth_name*RepeatTimes
        for frame_name_i in range(self.n_frames_per_clip):
            rgb_cache = Image.open(rgb_name[frame_name_i]).convert("RGB").resize((self.w, self.h), Image.BILINEAR)  # C * W * H
            rgb_cache = self.transform(rgb_cache)
            rgb[:, frame_name_i, :, :] = rgb_cache
            mask = Image.open(depth_name[frame_name_i]).convert("L").resize((self.w, self.h), Image.BILINEAR)  # C * W * H
            threshold = 10
            mask = mask.point(lambda p: p > threshold and 255)
            mask = self.mask_transform(mask)
            mask = torch.squeeze(mask)
            masks[frame_name_i, :, :] = mask
        return rgb, masks, label - 1

    def __len__(self):
        return int(self.sample_num)


def PadCollate(batch):
    rgbs = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    # max_length = max([ [*rgb.shape][1] for rgb in rgbs])
    # for i in range(len(rgbs)):
    #     frame_length = [*rgbs[i].shape][1]
    #     if frame_length < max_length:
    #         rgbs[i] = torch.cat((rgbs[i], 
    #                   torch.zeros([3, (int(max_length)-frame_length), 
    #                               [*rgbs[i].shape][2], [*rgbs[i].shape][3]],dtype=float)), 1)
    #         masks[i] = torch.cat((masks[i], 
    #                   torch.zeros([(int(max_length)-frame_length), 
    #                   [*rgbs[i].shape][2], [*rgbs[i].shape][3]],dtype=torch.long)), 0)
    return rgbs, masks, labels


# # trans = transforms.Compose([
# #                     # transforms.RandomResizedCrop(224, 126),
# #                     # transforms.CenterCrop(224),
# #                     #     transforms.RandomSizedCrop(255),
# #                     # transforms.RandomHorizontalFlip(),
# #                     transforms.ToTensor()
# #                     # transforms.Normalize([.5,.5,.5], [.5,.5,.5]),
# #                     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# #                 ])

# trans = transforms.Compose([transforms.ToTensor()])



# # tasks = ['Scene{:01}'.format(i) for i in range(1,7)]

# img_dataset = dataset_all(root_path, 'train1', n_frames_per_clip=16, 
#                                UnequalSequence = False, 
#                                img_size=(112, 112), stride=1,
#                                reverse=False, transform=trans,
#                                mask_trans = trans)
# rgb, depth, label = img_dataset.__getitem__(0)
# # print(set(img_dataset.labels))
# # print(len(set(img_dataset.labels)))



# # # # dataset_train_2 = dataset_all(root_path, 'train',
# # # #                                         n_frames_per_clip=1,
# # # #                                         UnequalSequence = False,
# # # #                                         img_size=(224, 126), stride=1,
# # # #                                         reverse=False, transform=trans,
# # # #                                         mask_trans=trans)
# # # # dataloader_train_2 = DataLoader(dataset_train_2, batch_size=64,
# # # #                                 shuffle=True, collate_fn = PadCollate,
# # # #                                 num_workers=8, pin_memory=True)
# # # # trainiter = iter(dataloader_train_2)
# # # # rgbs, masks, labels = trainiter.next()


# pdb.set_trace()



