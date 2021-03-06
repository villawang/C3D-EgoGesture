import pandas as pd
import os
import pdb
from tqdm import tqdm
import numpy as np


frame_path = '/home/zhengwei/dataset/egogesture/frames'
label_path = '/home/zhengwei/dataset/egogesture/labels-final-revised1'
annot_dict = {k: [] for k in ['rgb', 'depth', 'label']}

training_id = [3, 4, 5, 6, 8, 10, 15, 16, 17, 20, 21, 22, 23, 25, 26, 27, 30, 32, 36, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50]
validating_id = [1, 7, 12, 13, 24, 29, 33, 34, 35, 37]
testing_id = [2, 9, 11, 14, 18, 19, 28, 31, 41, 47]


for sub_i in tqdm(testing_id): # change to training_id and validating_id when saving to train and val
    frame_path_sub = os.path.join(frame_path, 'Subject{:02}'.format(sub_i))
    label_path_sub = os.path.join(label_path, 'subject{:02}'.format(sub_i))
    assert len([name for name in os.listdir(label_path_sub) if name != '.DS_Store']) == len([name for name in os.listdir(frame_path_sub)])
    for scene_i in range(1, len([name for name in os.listdir(frame_path_sub)])+1):
        rgb_path = os.path.join(frame_path_sub, 'Scene{:01}'.format(scene_i), 'Color')
        depth_path = os.path.join(frame_path_sub, 'Scene{:01}'.format(scene_i), 'Depth')
        label_path_iter = os.path.join(label_path_sub, 'Scene{:01}'.format(scene_i))
        assert len([name for name in os.listdir(label_path_iter) if 'csv'==name[-3::]]) == len([name for name in os.listdir(rgb_path)])
        assert len([name for name in os.listdir(label_path_iter) if 'csv'==name[-3::]]) == len([name for name in os.listdir(depth_path)])
        for group_i in range(1, len([name for name in os.listdir(rgb_path)])+1):
            rgb_path_group = os.path.join(rgb_path, 'rgb{:01}'.format(group_i))
            depth_path_group = os.path.join(depth_path, 'depth{:01}'.format(group_i))
            if os.path.isfile(os.path.join(label_path_iter, 'Group{:01}.csv'.format(group_i))):
                label_path_group = os.path.join(label_path_iter, 'Group{:01}.csv'.format(group_i))
            else:
                label_path_group = os.path.join(label_path_iter, 'group{:01}.csv'.format(group_i))
            # read the annotation files in the label path
            data_note = pd.read_csv(label_path_group, names = ['class', 'start', 'end'])
            data_note = data_note[np.isnan(data_note['start']) == False]
            for data_i in range(data_note.values.shape[0]):
                label = data_note.values[data_i,0]
                rgb = []
                depth = []
                for img_ind in range(int(data_note.values[data_i,1]), int(data_note.values[data_i,2]-1)):
                    rgb.append(os.path.join(rgb_path_group, '{:06}.jpg'.format(img_ind)))
                    depth.append(os.path.join(depth_path_group, '{:06}.jpg'.format(img_ind)))
                annot_dict['rgb'].append(rgb) 
                annot_dict['depth'].append(depth)
                annot_dict['label'].append(label)
annot_df = pd.DataFrame(annot_dict)

save_dir = '/home/zhengwei/dataset/egogesture'
# change here when using different train ids
# save_file = os.path.join(save_dir, 'train.pkl')
# save_file = os.path.join(save_dir, 'val.pkl')
save_file = os.path.join(save_dir, 'test.pkl')
annot_df.to_pickle(save_file)
