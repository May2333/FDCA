import re
import time
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm
from blip_model.blip_embs import  blip_embs
from data_utils import squarepad_transform, ComposedVideoDataset, targetpad_transform
from combiner import Combiner
from utils import extract_index_features, collate_fn, element_wise_sum, device


def compute_finecvr_val_metrics(relative_val_dataset: ComposedVideoDataset, clip_model: CLIP, index_features: torch.tensor,
                             index_names: List[str], combining_function: callable, combiner) -> Tuple[
    float, float, float, float, float, float, float]:
    """
    Compute validation metrics on FineCVR dataset
    :param relative_val_dataset: finecvr validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """
    # Generate predictions
    predicted_features, reference_names, target_names = \
        generate_finecvr_val_predictions(clip_model, relative_val_dataset, combining_function, index_names, index_features)

    with torch.no_grad():
        b = 128
        index_features_tmp = []
        for bt_index in range(int(len(index_features) / b) + 1):
            if bt_index < int(len(index_features) / b):
                index_features_tmp.append(combiner.time_process(index_features[bt_index * b:(bt_index + 1) * b, ]))

            else:
                index_features_tmp.append(combiner.time_process(index_features[bt_index * b:, ]))
        index_features = torch.cat(index_features_tmp, dim=0)

    index_features = F.normalize(index_features, dim=-1).float()
    predicted_features = predicted_features
    print("Compute the distances and sort the results")

    print("Compute finecvr validation metrics")
    b = 32
    sorted_indices_tmp = []
    sorted_index_names = []
    labels = []
    for bt_index in tqdm(range(int(len(predicted_features) / b) + 1)):
        if bt_index < int(len(predicted_features) / b):
            tmp = 1 - predicted_features[bt_index * b:(bt_index + 1) * b, ] @ index_features.T # [32,all_data_n]
            sorted_indices_tmp = torch.argsort(tmp.cpu(), dim=-1) #[32, all_data_n]
            sorted_index_names_tmp = torch.tensor(index_names)[sorted_indices_tmp] #[32, all_data_n]
            reference_mask_tmp = torch.tensor( # [32, all_data_n]![32, all_data_n]
                sorted_index_names_tmp != torch.tensor(reference_names[bt_index * b:(bt_index + 1) * b]).unsqueeze(
                    dim=1).repeat(1, len(index_names)).reshape(len(target_names[bt_index * b:(bt_index + 1) * b]), -1))

            sorted_index_names_tmp = sorted_index_names_tmp[reference_mask_tmp].reshape(sorted_index_names_tmp.shape[0],
                                                                                        sorted_index_names_tmp.shape[
                                                                                            1] - 1)#[32, all_data_n-1]
            labels_tmp = torch.tensor(
                sorted_index_names_tmp[:, :50] == torch.tensor(target_names[bt_index * b:(bt_index + 1) * b]).unsqueeze(
                    dim=1).repeat(1, 50).reshape(
                    len(target_names[bt_index * b:(bt_index + 1) * b]), -1))

        else:
            tmp = 1 - predicted_features[bt_index * b:, ] @ index_features.T
            #         distances_tmp.append(tmp.cpu())
            sorted_indices_tmp = torch.argsort(tmp.cpu(), dim=-1)
            sorted_index_names_tmp = torch.tensor(index_names)[sorted_indices_tmp]
            reference_mask_tmp = torch.tensor(
                sorted_index_names_tmp != torch.tensor(reference_names[bt_index * b:]).unsqueeze(dim=1).repeat(1,
                                                                                                               len(index_names)).reshape(
                    len(target_names[bt_index * b:]), -1))
            sorted_index_names_tmp = sorted_index_names_tmp[reference_mask_tmp].reshape(sorted_index_names_tmp.shape[0],
                                                                                        sorted_index_names_tmp.shape[
                                                                                            1] - 1)
            labels_tmp = torch.tensor(
                sorted_index_names_tmp[:, :50] == torch.tensor(target_names[bt_index * b:]).unsqueeze(dim=1).repeat(1,
                                                                                                                    50).reshape(
                    len(target_names[bt_index * b:]), -1))
        labels.append(labels_tmp)
        sorted_index_names.append(sorted_index_names_tmp)
    labels = torch.cat(labels, dim=0)
    sorted_index_names = torch.cat(sorted_index_names, dim=0)

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = -1
    group_recall_at2 = -1
    group_recall_at3 = -1

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50

def generate_finecvr_val_predictions(clip_model: CLIP, relative_val_dataset: ComposedVideoDataset,
                                  combining_function: callable, index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str], List[str], List[List[str]]]:
    """
    Compute finecvr predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: finecvr validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features, reference names, target names and group members
    """
    print("Compute finecvr validation predictions")
    clip_model.eval()

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=1024, num_workers=4,
                                     pin_memory=True, collate_fn=collate_fn)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features, target_names, group_members and reference_names
    predicted_features = torch.empty((0, clip_model.text_proj.out_features)).to(device, non_blocking=True)
    target_names = []
    reference_names = []
    #
    for batch_reference_names, batch_target_names, (captions, captions_wo_negation), batch_group_members, middle_feature in tqdm(relative_val_loader):  # Load data
        middle_feature = middle_feature.to(device, non_blocking=True).float()
        # Compute the predicted features
        with torch.no_grad():
            if len(name_to_feat) == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names.numpy())(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features,_ = combining_function((reference_image_features, middle_feature), (captions, captions_wo_negation))

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)
        reference_names.extend(batch_reference_names)
    return predicted_features, reference_names, target_names


def finecvr_val_retrieval(combining_function: callable, clip_model: CLIP, preprocess: callable, args, combiner):
    """
    Perform retrieval on finecvr validation set computing the metrics. To combine the features the `combining_function`
    is used
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param clip_model: CLIP model
    :param preprocess: preprocess pipeline
    """

    clip_model = clip_model.float().eval()

    # Define the validation datasets and extract the index features
    classic_val_dataset = ComposedVideoDataset('test', 'classic', preprocess, args.data_pth, args.dataset_op)
    index_features, index_names = extract_index_features(classic_val_dataset, clip_model)
    relative_val_dataset = ComposedVideoDataset('test', 'relative', preprocess, args.data_pth, args.dataset_op)

    return compute_finecvr_val_metrics(relative_val_dataset, clip_model, index_features, index_names,
                                    combining_function, combiner)
def get_blip_config(model="base"):
    config = dict()
    if model == "base":
        config[
            "pretrained"
        ] = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth "
        config["vit"] = "base"
        config["batch_size_train"] = 32
        config["batch_size_test"] = 16
        config["vit_grad_ckpt"] = True
        config["vit_ckpt_layer"] = 4
        config["init_lr"] = 1e-5
    elif model == "large":
        config[
            "pretrained"
        ] = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth"
        config["vit"] = "large"
        config["batch_size_train"] = 16
        config["batch_size_test"] = 32
        config["vit_grad_ckpt"] = True
        config["vit_ckpt_layer"] = 12
        config["init_lr"] = 5e-6

    config["image_size"] = 384
    config["queue_size"] = 57600
    config["alpha"] = 0.4
    config["k_test"] = 256
    config["negative_all_rank"] = True

    return config

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", default='finecvr', type=str, help="should be either 'finecvr'")
    parser.add_argument("--data_pth", type=str, default="FineCVR",
                        help="the dataset's path")
    parser.add_argument("--combining-function", type=str, default='combiner',
                        help="Which combining function use, should be in ['combiner', 'sum']")
    parser.add_argument("--combiner-path", type=Path,  default="combiner_arithmetic_3.pt", help="path to trained Combiner")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--clip-model-name", default="ViT-B/32", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path",type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")

    args = parser.parse_args()
    print("Creating blip model")
    config = get_blip_config('large')
    blip_model = blip_embs(
        pretrained=config["pretrained"],
        image_size=config["image_size"],
        vit=config["vit"],
        vit_grad_ckpt=config["vit_grad_ckpt"],
        vit_ckpt_layer=config["vit_ckpt_layer"],
        queue_size=config["queue_size"],
        negative_all_rank=config["negative_all_rank"],
    )

    blip_model.eval()
    input_dim = blip_model.text_proj.out_features
    feature_dim = blip_model.text_proj.out_features
    clip_preprocess = None
    clip_model = blip_model.float()

    if args.clip_model_path:
        print('Trying to load the CLIP model')
        saved_state_dict = torch.load(args.clip_model_path, map_location=device)
        clip_model.load_state_dict(saved_state_dict["CLIP"])
        print('CLIP model loaded successfully')

    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    elif args.transform == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)
    else:
        print('CLIP default preprocess pipeline is used')
        preprocess = clip_preprocess

    if args.combining_function.lower() == 'sum':
        if args.combiner_path:
            print("Be careful, you are using the element-wise sum as combining_function but you have also passed a path"
                  " to a trained Combiner. Such Combiner will not be used")
        combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim, clip_model=clip_model).to(device, non_blocking=True)
        state_dict = torch.load(args.combiner_path, map_location=device)
        combiner.load_state_dict(state_dict["Combiner"])
        combiner.eval()
        combining_function = element_wise_sum

    elif args.combining_function.lower() == 'combiner':
        combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim, clip_model=clip_model).to(device, non_blocking=True)
        state_dict = torch.load(args.combiner_path, map_location=device)
        combiner.load_state_dict(state_dict["Combiner"])
        combiner.eval()
        combining_function = combiner.combine_features
    else:
        raise ValueError("combiner_path should be in ['sum', 'combiner']")

    if args.dataset.lower() == 'finecvr':
        group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = \
            finecvr_val_retrieval(combining_function, clip_model, preprocess, args, combiner)

        print(f"{group_recall_at1 = }")
        print(f"{group_recall_at2 = }")
        print(f"{group_recall_at3 = }")
        print(f"{recall_at1 = }")
        print(f"{recall_at5 = }")
        print(f"{recall_at10 = }")
        print(f"{recall_at50 = }")

    
    else:
        raise ValueError("Dataset should be 'finecvr'")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print("time cost", time.time()-t0)