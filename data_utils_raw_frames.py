import json
import os.path
import random
from pathlib import Path
from typing import List

import numpy
import torch
import PIL
import PIL.Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import csv
import pandas as pd
import numpy as np
import time
import h5py


base_path = Path(__file__).absolute().parents[1].absolute()


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class FineCVRDataset(Dataset):
    """
       CIRR dataset class which manage CIRR data
       The dataset can be used in 'relative' or 'classic' mode:
           - In 'classic' mode the dataset yield tuples made of (image_name, image)
           - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, rel_caption) when split == train
                - (reference_name, target_name, rel_caption, group_members) when split == val
                - (pair_id, reference_name, rel_caption, group_members) when split == test1
    """

    def __init__(self, split: str, mode: str, preprocess: callable, dataset_pth):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                  - In 'classic' mode the dataset yield tuples made of (image_name, image)
                  - In 'relative' mode the dataset yield tuples made of:
                        - (reference_image, target_image, rel_caption) when split == train
                        - (reference_name, target_name, rel_caption, group_members) when split == val
                        - (pair_id, reference_name, rel_caption, group_members) when split == test1
        :param preprocess: function which preprocesses the image
        """
        self.preprocess = preprocess
        self.mode = mode
        self.split = split
        self.num = 0
        self.num_frames = 8
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        self.dataset_root = dataset_pth
        self.train_dataset = pd.read_table(os.path.join(self.dataset_root, 'annotations', 'train_remaped.txt'), names=['idx', 'ref', 'target', 'cap','source','class_id'], header=None, quoting=csv.QUOTE_NONE)
        self.val_dataset = pd.read_table(os.path.join(self.dataset_root, 'annotations', 'test_remaped.txt'), names=['idx', 'ref', 'target', 'cap','source','class_id'], header=None, quoting=csv.QUOTE_NONE)

        self.val_member_dict = dict(self.val_dataset.groupby(['ref', 'cap']).groups)

        for run_tp in ['train', 'val']:
            for data_tp in ['ref', 'cap', 'target', 'source']:
                setattr(self, "{}_{}_list".format(run_tp, data_tp), getattr(self,"{}_dataset".format(run_tp))[data_tp].tolist())

        if self.split == 'val':
            id2vdoname_pt = os.path.join(self.dataset_root, 'annotations', "id2vdoname_test.json")
        else:
            id2vdoname_pt = os.path.join(self.dataset_root, 'annotations', "id2vdoname_train.json")
        with open(id2vdoname_pt, 'r', encoding='utf-8') as file:
            content = file.read()
            self.id2vdoname = json.loads(content)
            
        self.feature_path = os.path.join(self.dataset_root, "embbedings/CLIP_RN50x4_high_8_640")
        print(f"FineCVR {split} dataset in {mode} mode initialized")

    def _get_8_imgs_from_id(self, vdo, frames_path):
        vdo_name = self.id2vdoname[str(vdo)]
        pth = os.path.join(frames_path, vdo_name)
        if not os.path.exists(pth):
            if 'hvu' in frames_path:
                youtube_id = vdo_name[:11]
                time_start = int(vdo_name.split('_')[-2])
                time_end = int(vdo_name.split('_')[-1])
                vdo_name = "{}_%06d_%06d".format(youtube_id) % (time_start, time_end)
                if not os.path.exists(os.path.join(frames_path, vdo_name)):
                    vdo_name = "{}_{}_{}".format(youtube_id, time_start, time_end)
                if not os.path.exists(os.path.join(frames_path, vdo_name)):
                    vdo_name = "{}_%.1f_%.1f".format(youtube_id) % (time_start, time_end)
                if not os.path.exists(os.path.join(frames_path, vdo_name)):
                    vdo_name = "{}_%.1f_%.1f".format(youtube_id) % (time_start + 0.5, time_end + 0.5)
                if not os.path.exists(os.path.join(frames_path, vdo_name)):
                    vdo_name = "{}_%.1f_%.1f".format(youtube_id) % (time_start - 0.5, time_end - 0.5)
                if not os.path.exists(os.path.join(frames_path, vdo_name)):
                    vdo_name = "{}".format(youtube_id)
                pth = os.path.join(frames_path, vdo_name)
        imgs_list = sorted(os.listdir(pth))
        if len(imgs_list)<self.num_frames:
            vlen = len(imgs_list)
            frame_indices = list(range(vlen)) + [vlen-1] *(self.num_frames - vlen)
        else:
            frame_indices = np.arange(0, self.num_frames, dtype=int)
        imgs = []
        for i, img_i in enumerate(frame_indices):
            img = imgs_list[img_i]
            imgs.append(self._get_img_from_path(os.path.join(pth, img), self.preprocess))
        imgs = torch.stack(imgs, dim=0)
        return imgs

    def _get_img_from_path(self, img_path, transform=None):
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            if transform is not None:
                time0=time.time()
                img = transform(img)
        return img

    def _get_img_from_id(self, vdo):
        imgs = self._get_8_imgs_from_id(vdo)
        return imgs

    def __getitem__(self, index):
        # try:
        if self.mode == 'relative':

            if self.split == 'train':
                reference_vdo = self.train_ref_list[index]
                reference_fea = np.load(os.path.join(self.feature_path, self.id2vdoname[str(reference_vdo)] + '.npy'))
                with h5py.File(os.path.join(self.middle_feature_path, self.id2vdoname[str(reference_vdo)] + '.h5'), 'r') as f:
                    reference_middle_fea =  np.array(f['middle_layer_feature'])

                rel_caption = self.train_cap_list[index]

                target_vdo = self.train_target_list[index]
                target_fea = np.load(os.path.join(self.feature_path, self.id2vdoname[str(target_vdo)] + '.npy'))
                with h5py.File(os.path.join(self.middle_feature_path, self.id2vdoname[str(target_vdo)] + '.h5'), 'r') as f:
                    target_middle_fea =  np.array(f['middle_layer_feature'])
                return (reference_fea, reference_middle_fea), (target_fea, target_middle_fea), rel_caption

            elif self.split == 'val':
                reference_name = self.val_ref_list[index]
                rel_caption = self.val_cap_list[index]
                target_hard_name = self.val_target_list[index]
                group_members = np.array(self.val_target_list)[self.val_member_dict[(self.val_ref_list[index], self.val_cap_list[index])].values]
                with h5py.File(os.path.join(self.middle_feature_path, self.id2vdoname[str(reference_name)] + '.h5'), 'r') as f:
                    ref_vdo_fea_middle = np.array(f['middle_layer_feature'])
                return reference_name, target_hard_name, rel_caption, [group_members[0]], ref_vdo_fea_middle

        elif self.mode == 'classic':
            vdo_name = self.id2vdoname[str(index)]
            if 'video' in vdo_name:
                frames_path = os.path.join("finecvr_raw_frames", "msrvtt_frames")
            elif len(vdo_name)==5:
                frames_path = os.path.join("finecvr_raw_frames", "ag_frames")
            elif len(vdo_name)>13 and len(vdo_name)<17:
                frames_path = os.path.join("finecvr_raw_frames", "an_frames")
            else:
                frames_path = os.path.join("finecvr_raw_frames", "hvu_frames")
            imgs = self._get_8_imgs_from_id(index, frames_path)
            return index, imgs, frames_path

        else:
            raise ValueError("mode should be in ['relative', 'classic']")


    def __len__(self):
        if self.mode == 'relative':
            return len(getattr(self, "{}_dataset".format(self.split)))
        elif self.mode == 'classic':
            return len(self.id2vdoname)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")