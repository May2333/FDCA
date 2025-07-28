import os

from comet_ml import Experiment
import json
import multiprocessing
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, harmonic_mean, geometric_mean
from typing import List
import clip
import numpy as np
import pandas as pd
import torch
import random
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import base_path, squarepad_transform, targetpad_transform, ComposedVideoDataset
from combiner import Combiner
from utils import collate_fn, update_train_running_results, set_train_bar_description, save_model, \
    extract_index_features,  device
from validate import compute_finecvr_val_metrics
from blip_model.blip_embs import  blip_embs

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(2023)

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

def combiner_training_composedvideo(projection_dim: int, hidden_dim: int, num_epochs: int, clip_model_name: str,
                           combiner_lr: float, batch_size: int, clip_bs: int, validation_frequency: int, transform: str,
                           save_training: bool, save_best: bool, **kwargs):
    """
    Train the Combiner on FineCVR dataset keeping frozen the CLIP model
    :param projection_dim: Combiner projection dimension
    :param hidden_dim: Combiner hidden dimension
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param combiner_lr: Combiner learning rate
    :param batch_size: batch size of the Combiner training
    :param clip_bs: batch size of the CLIP feature extraction
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the Combiner network
    :param save_best: when True save only the weights of the best Combiner wrt three different averages of the metrics
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio` as kwarg. If you want to load a
                fine-tuned version of clip you should provide `clip_model_path` as kwarg.
    """

    training_path: Path = Path(
        f"models/{clip_model_name}/{args.save_name}")
    if not os.path.exists(training_path):
        os.makedirs(training_path)

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

    if transform == "clip" or transform == "blip":
        preprocess = clip_preprocess
        print('CLIP default preprocess pipeline is used')
    elif transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio: } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")


    blip_model = blip_model.float()

    # Define the validation datasets and extract the validation index features
    relative_val_dataset = ComposedVideoDataset('val', 'relative', preprocess, dataset_pth=args.data_pth)
    classic_val_dataset = ComposedVideoDataset('val', 'classic', preprocess, dataset_pth=args.data_pth)
    val_index_features, val_index_names = extract_index_features(classic_val_dataset, blip_model)

    # Define the combiner and the train dataset
    combiner = Combiner(feature_dim, projection_dim, hidden_dim, clip_model=blip_model).to(device, non_blocking=True)
    relative_train_dataset = ComposedVideoDataset('train', 'relative', preprocess, dataset_pth=args.data_pth)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size, num_workers=4,
                                       pin_memory=True, collate_fn=collate_fn, drop_last=True, shuffle=True)

    # Save all the hyperparameters on a file
    training_hyper_params['model'] = str(combiner)
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.Adam(combiner.parameters(), lr=combiner_lr)
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best results to zero
    if save_best:
        best_arithmetic = -100

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')
    combiner.clip_model.requires_grad = False
    for epoch in range(num_epochs):
        with experiment.train():
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            combiner.train()
            train_bar = tqdm(relative_train_loader, ncols=150)
            for idx, ((reference_feas, reference_middle_feas), (target_feas, target_middle_feas), (captions, captions_wo_neg)) in enumerate(train_bar):  # Load a batch of triplets
                images_in_batch = reference_feas.size(0)
                step = len(train_bar) * epoch + idx

                optimizer.zero_grad()

                reference_feas = reference_feas.to(device, non_blocking=True)
                reference_middle_feas = reference_middle_feas.to(device, non_blocking=True)
                target_feas = target_feas.to(device, non_blocking=True)
                target_middle_feas = target_middle_feas.to(device, non_blocking=True)

                # Compute the logits and loss
                with torch.cuda.amp.autocast():
                    logits, token_logist, logits_2, remained_loss, remained_loss_2, triplet_loss = combiner((reference_feas, reference_middle_feas), (captions, captions_wo_neg), (target_feas, target_middle_feas))
                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    loss = crossentropy_criterion(logits, ground_truth)
                    loss_token = crossentropy_criterion(token_logist, ground_truth)
                    loss_2 = crossentropy_criterion(logits_2, ground_truth)
                    loss = loss + loss_token+ loss_2 + remained_loss + remained_loss_2+triplet_loss

                # Backpropagate and update the weights
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                experiment.log_metric('similar_per', 0, step=step)
                train_running_results['similar_per'] = 0
                update_train_running_results(train_running_results, loss, images_in_batch)
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)
            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            # Training CSV logging
            training_log_frame = pd.concat(
                [training_log_frame,
                 pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
            training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            blip_model = blip_model.float()  # In validation we use fp32 CLIP model
            with experiment.validate():
                with torch.no_grad():
                    combiner.eval()

                    # Compute and log validation metrics
                    results = compute_finecvr_val_metrics(relative_val_dataset, blip_model, val_index_features,
                                                       val_index_names, combiner.combine_features, combiner)
                    group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results
                    print(results)
                    results_dict = {
                        'group_recall_at1': group_recall_at1,
                        'group_recall_at2': group_recall_at2,
                        'group_recall_at3': group_recall_at3,
                        'recall_at1': recall_at1,
                        'recall_at5': recall_at5,
                        'recall_at10': recall_at10,
                        'recall_at50': recall_at50,
                        'mean(R@5+R_s@1)': (group_recall_at1 + recall_at5) / 2,
                        'arithmetic_mean': mean(results),
                    }

                    print(json.dumps(results_dict, indent=4))
                    experiment.log_metrics(
                        results_dict,
                        epoch=epoch
                    )

                    # Validation CSV logging
                    log_dict = {'epoch': epoch}
                    log_dict.update(results_dict)
                    validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                    validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

                    # Save model
                    if save_training:
                        if save_best and results_dict['recall_at50'] > best_arithmetic:
                            best_arithmetic = results_dict['recall_at50']
                            save_model('combiner_arithmetic', epoch, combiner, training_path)
                        if not save_best:
                            save_model(f'combiner_{epoch}', epoch, combiner, training_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='finecvr', help="should be 'finecvr'")
    parser.add_argument("--data_pth", type=str, default="FineCVR",
                        help="the dataset's path")
    parser.add_argument("--api-key", type=str, help="api for Comet logging")
    parser.add_argument("--workspace", type=str, help="workspace of Comet logging")
    parser.add_argument("--experiment-name", type=str, help="name of the experiment on Comet")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--num-epochs", default=5, type=int, help="number training epochs")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=str, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--combiner-lr", default=1e-4, type=float, help="Combiner learning rate")
    parser.add_argument("--batch-size", default=1024, type=int, help="Batch size of the Combiner training")
    parser.add_argument("--clip-bs", default=32, type=int, help="Batch size during CLIP feature extraction")
    parser.add_argument("--validation-frequency", default=3, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--save-training", dest="save_training", action='store_true',
                        help="Whether save the training model")
    parser.add_argument("--save-best", default=True, action='store_true',
                        help="Save only the best model during training")
    parser.add_argument("--save_name", default='iclr/blip', action='store_true',
                        help="Save only the best model during training")

    args = parser.parse_args()
    if args.dataset.lower() not in ['finecvr']:
        raise ValueError("Dataset should be finecvr")

    training_hyper_params = {
        "projection_dim": args.projection_dim,
        "hidden_dim": args.hidden_dim,
        "num_epochs": args.num_epochs,
        "clip_model_name": args.clip_model_name,
        "clip_model_path": args.clip_model_path,
        "combiner_lr": args.combiner_lr,
        "batch_size": args.batch_size,
        "clip_bs": args.clip_bs,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
    }

    if args.api_key and args.workspace:
        print("Comet logging ENABLED")
        experiment = Experiment(
            api_key=args.api_key,
            project_name=f"{args.dataset} combiner training",
            workspace=args.workspace,
            disabled=False
        )
        if args.experiment_name:
            experiment.set_name(args.experiment_name)
    else:
        print("Comet loging DISABLED, in order to enable it you need to provide an api key and a workspace")
        experiment = Experiment(
            api_key="",
            project_name="",
            workspace="",
            disabled=True
        )

    experiment.log_code(folder=str(base_path / 'src'))
    experiment.log_parameters(training_hyper_params)

    if args.dataset.lower() == 'finecvr':
        combiner_training_composedvideo(**training_hyper_params)
