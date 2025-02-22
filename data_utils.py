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

base_path = Path(__file__).absolute().parents[1].absolute()


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
       CIRR dataset class which manage CIRR data
       The dataset can be used in 'relative' or 'classic' mode:
           - In 'classic' mode the dataset yield tuples made of (image_name, image)
           - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, rel_caption) when split == train
                - (reference_name, target_name, rel_caption, group_members) when split == val
                - (pair_id, reference_name, rel_caption, group_members) when split == test1
    """

    def __init__(self, split: str, mode: str, preprocess: callable, dataset_pth, dataset_op):
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

        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        # get triplets made by (reference_image, target_image, relative caption)
        self.dataset_root = dataset_pth
        self.train_dataset = pd.read_table(os.path.join(self.dataset_root, 'modified_dataset', dataset_op, 'vdo_modified_text_train_clip_cleaned_remaped.txt'), names=['idx', 'ref', 'target', 'cap','source','class_id'], header=None, quoting=csv.QUOTE_NONE)
        self.val_dataset = pd.read_table(os.path.join(self.dataset_root, 'modified_dataset', dataset_op, 'vdo_modified_text_val_clip_cleaned_remaped.txt'), names=['idx', 'ref', 'target', 'cap','source','class_id'], header=None, quoting=csv.QUOTE_NONE)
        self.test_dataset = pd.read_table(os.path.join(self.dataset_root, 'modified_dataset', dataset_op, 'vdo_modified_text_test_clip_cleaned_remaped.txt'), names=['idx', 'ref', 'target', 'cap','source','class_id'], header=None, quoting=csv.QUOTE_NONE)

        self.val_member_dict = dict(self.val_dataset.groupby(['ref', 'cap']).groups)
        self.test_member_dict = dict(self.test_dataset.groupby(['ref', 'cap']).groups)

        for run_tp in ['train', 'val', 'test']:
            for data_tp in ['ref', 'cap', 'target', 'source']:
                setattr(self, "{}_{}_list".format(run_tp, data_tp), getattr(self,"{}_dataset".format(run_tp))[data_tp].tolist())

        # get a mapping from image name to relative path
        if self.split == 'test':
            id2vdoname_pt = os.path.join(self.dataset_root, 'modified_dataset', dataset_op, "id2vdoname_test.json")
        else:
            id2vdoname_pt = os.path.join(self.dataset_root, 'modified_dataset', dataset_op, "id2vdoname_train_val.json")
        with open(id2vdoname_pt, 'r', encoding='utf-8') as file:
            content = file.read()
            self.id2vdoname = json.loads(content)

        # self.feature_path = os.path.join(self.dataset_root, "action_genome_feature_CLIP_Res50x4_8frames_high/")

        # self.ag_feature_path = os.path.join(self.dataset_root, "..", "action_genome_v1.0",
        #                                     "complete_action_genome_feature_CLIP_Res50x4_8frames_high/")
        # self.ag_middle_feature_path = os.path.join(self.dataset_root, "..", "action_genome_v1.0",
        #                                            "complete_action_genome_feature_CLIP_Res50x4_8frames_middle_16_640/")
        # self.an_feature_path = os.path.join(self.dataset_root, "..", "composed_activity_data",
        #                                     "an_feature_CLIP_Res50x4_8frames_high/")
        # self.an_middle_feature_path = os.path.join(self.dataset_root, "..", "composed_activity_data",
        #                                            "an_feature_CLIP_Res50x4_8frames_middle_16_640/")
        self.feature_path = os.path.join("/dev/shm/CLIP_RN50x4_high_8_640")
        print(f"action genome {split} dataset in {mode} mode initialized")
#         print(f"action genome {split} dataset in {mode} mode initialized")

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

                    # with h5py.File(os.path.join(feature_path, self.id2vdoname[str(reference_vdo)] + '.h5'), 'r') as f:
                    #     reference_fea =  f['high_feature'][()]

                    reference_middle_fea = np.zeros_like(reference_fea)
                    target_middle_fea = reference_middle_fea
                    # with h5py.File(os.path.join(middle_feature_path, self.id2vdoname[str(reference_vdo)] + '.h5'), 'r') as f:
                    #     reference_middle_fea =  np.array(f['middle_layer_feature'])
                    rel_caption = self.train_cap_list[index]

                    target_vdo = self.train_target_list[index]
                    target_fea=  np.load(os.path.join(self.feature_path, self.id2vdoname[str(target_vdo)] + '.npy'))
                    # with h5py.File(os.path.join(feature_path, self.id2vdoname[str(target_vdo)] + '.h5'), 'r') as f:
                    #     target_fea =  f['high_feature'][()]
                    # target_fea = np.load(os.path.join(self.feature_path, self.id2vdoname[str(target_vdo)]) + '.npy')
                    # with h5py.File(os.path.join(middle_feature_path, self.id2vdoname[str(target_vdo)] + '.h5'), 'r') as f:
                    #     target_middle_fea =  np.array(f['middle_layer_feature'])

                    return (reference_fea, reference_fea), (target_fea, target_fea), (rel_caption, self.remove_neg(rel_caption))

                elif self.split == 'val':
                    vdo_source = self.val_source_list[index]
#                     if vdo_source == 'an':
#                         feature_path = self.an_feature_path
#                         middle_feature_path = self.an_middle_feature_path
#                     else:
#                         feature_path = self.ag_feature_path
#                         middle_feature_path = self.ag_middle_feature_path
                    reference_name = self.val_ref_list[index]
                    rel_caption = self.val_cap_list[index]
                    target_hard_name = self.val_target_list[index]
                    # group_members = np.array(self.val_target_list)[self.val_member_dict[(self.val_ref_list[index], self.val_cap_list[index])].values]
                    # with h5py.File(os.path.join(middle_feature_path, self.id2vdoname[str(reference_name)] + '.h5'), 'r') as f:
                    #     ref_vdo_fea_middle = np.array(f['middle_layer_feature'])
                    ref_vdo_fea_middle = 0
                    return reference_name, target_hard_name, (rel_caption, self.remove_neg(rel_caption)), target_hard_name, ref_vdo_fea_middle

                elif self.split == 'test':
                    vdo_source = self.test_source_list[index]
#                     if vdo_source == 'an':
#                         feature_path = self.an_feature_path
#                         middle_feature_path = self.an_middle_feature_path
#                     else:
#                         feature_path = self.ag_feature_path
#                         middle_feature_path = self.ag_middle_feature_path
                    pair_id = index
                    reference_name = self.test_ref_list[index]
                    rel_caption = self.test_cap_list[index]
                    target_hard_name = self.test_target_list[index]
                    # group_members = np.array(self.test_target_list)[self.test_member_dict[(self.test_ref_list[index], self.test_cap_list[index])].values]
                    # with h5py.File(os.path.join(middle_feature_path, self.id2vdoname[str(reference_name)] + '.h5'), 'r') as f:
                    #     ref_vdo_fea_middle = np.array(f['middle_layer_feature'])
                    ref_vdo_fea_middle = 0
                    return reference_name, target_hard_name, (rel_caption, self.remove_neg(rel_caption)), target_hard_name, ref_vdo_fea_middle

            elif self.mode == 'classic':
                vdo_name = self.id2vdoname[str(index)]
#                 if len(vdo_name)>5:
#                     feature_path = self.an_feature_path
#                     middle_feature_path = self.an_middle_feature_path
#                 else:
#                     feature_path = self.ag_feature_path
#                     middle_feature_path = self.ag_middle_feature_path
                vdo_fea = np.load(os.path.join(self.feature_path, str(vdo_name) + '.npy'))

                # with h5py.File(os.path.join(feature_path, vdo_name + '.h5'),
                #                'r') as f:
                #     vdo_fea = f['high_feature'][()]
                # vdo_fea = np.load(os.path.join(self.feature_path, vdo_name + '.npy'))
                # im = PIL.Image.open(image_path)
                # image = self.preprocess(im)
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
            # return 99
        else:
            raise ValueError("mode should be in ['relative', 'classic']")