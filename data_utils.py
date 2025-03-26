import json
from pathlib import Path
from typing import List

import PIL
import PIL.Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import csv
import pandas as pd
import numpy as np
import os
import h5py

base_path = Path(__file__).absolute().parents[0].absolute()


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class SquarePad:
    """
    Square pad the input image with zero padding
    """

    def __init__(self, size: int):
        """
        For having a consistent preprocess pipeline with CLIP we need to have the preprocessing output dimension as
        a parameter
        :param size: preprocessing output dimension
        """
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


def squarepad_transform(dim: int):
    """
    CLIP-like preprocessing transform on a square padded image
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def targetpad_transform(target_ratio: float, dim: int):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class ComposedVideoDataset(Dataset):
    """
       FineCVR dataset class which manage FineCVR data
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
        :param dataset_pth: dataset path root
        """
        self.preprocess = preprocess
        self.mode = mode
        self.split = split

        if split not in ['train', 'val']:
            raise ValueError("split should be in ['train', 'val']")
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


    def remove_neg(self, cap):
        if 'instead of' in cap:
            cap = cap.replace("instead of", "")
        if 'not' in cap:
            cap = cap.replace("not", "")
        if 'rather than' in cap:
            cap = cap.replace("rather than", "")
        return cap

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':

                if self.split == 'train':
                    reference_vdo = self.train_ref_list[index]
                    vdo_source = self.train_source_list[index]
                    reference_fea = np.load(os.path.join(self.feature_path, self.id2vdoname[str(reference_vdo)] + '.npy'))
                    
                    rel_caption = self.train_cap_list[index]

                    target_vdo = self.train_target_list[index]
                    target_fea=  np.load(os.path.join(self.feature_path, self.id2vdoname[str(target_vdo)] + '.npy'))

                    return (reference_fea, reference_fea), (target_fea, target_fea), (rel_caption, self.remove_neg(rel_caption))

                elif self.split == 'val':
                    vdo_source = self.val_source_list[index]
                    reference_name = self.val_ref_list[index]
                    rel_caption = self.val_cap_list[index]
                    target_hard_name = self.val_target_list[index]
                    ref_vdo_fea_middle = 0
                    return reference_name, target_hard_name, (rel_caption, self.remove_neg(rel_caption)), target_hard_name, ref_vdo_fea_middle

            elif self.mode == 'classic':
                vdo_name = self.id2vdoname[str(index)]
                vdo_fea = np.load(os.path.join(self.feature_path, str(vdo_name) + '.npy'))
                return index, vdo_fea

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(getattr(self, "{}_dataset".format(self.split)))
        elif self.mode == 'classic':
            return len(self.id2vdoname)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")