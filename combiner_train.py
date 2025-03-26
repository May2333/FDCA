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
    extract_index_features, generate_randomized_fiq_caption, device
from validate import compute_val_metrics
from pytorch_transformers import BertModel, BertConfig, BertTokenizer
from model.clip import _transform, load

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(2023)


def combiner_training_composedvideo(projection_dim: int, hidden_dim: int, num_epochs: int, clip_model_name: str,
                           combiner_lr: float, batch_size: int, clip_bs: int, validation_frequency: int, transform: str,
                           save_training: bool, save_best: bool, **kwargs):
    """
    Train the Combiner on dataset keeping frozen the CLIP model
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

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/version_11/{clip_model_name}/{args.save_name}")
    if not os.path.exists(training_path):
        os.makedirs(training_path)



    clip_model, clip_preprocess, _= load(clip_model_name, device=device, jit=False)

    clip_model.eval()
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim


    if transform == "clip":
        preprocess = clip_preprocess
        print('CLIP default preprocess pipeline is used')
    elif transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    if kwargs.get("clip_model_path"):
        print('Trying to load the fine-tuned CLIP model')
        clip_model_path = kwargs["clip_model_path"]
        state_dict = torch.load(clip_model_path, map_location=device)
        clip_model.load_state_dict(state_dict["CLIP"])
        print('CLIP model loaded successfully')

    clip_model = clip_model.float()

    # Define the validation datasets and extract the validation index features
    relative_val_dataset = ComposedVideoDataset('val', 'relative', preprocess, dataset_pth=args.data_pth, dataset_op=args.dataset_op)
    classic_val_dataset = ComposedVideoDataset('val', 'classic', preprocess, dataset_pth=args.data_pth, dataset_op=args.dataset_op)
    val_index_features, val_index_names = extract_index_features(classic_val_dataset, clip_model)

    # Define the combiner and the train dataset
    combiner = Combiner(feature_dim, projection_dim, hidden_dim, clip_model=clip_model).to(device, non_blocking=True)
    relative_train_dataset = ComposedVideoDataset('train', 'relative', preprocess, dataset_pth=args.data_pth, dataset_op=args.dataset_op)
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
        best_harmonic = 0
        best_geometric = 0
        best_arithmetic = -100

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    lossmetric = nn.MSELoss()
    # Start with the training loop
    print('Training loop started')
    combiner.clip_model.requires_grad = False
    for epoch in range(num_epochs):
        if torch.cuda.is_available():  
            clip.model.convert_weights(clip_model)  # Convert CLIP model in fp16 to reduce computation and memory
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
                text_inputs = clip.tokenize(captions, truncate=True).to(device, non_blocking=True)
                text_inputs_wo_neg = clip.tokenize(captions_wo_neg, truncate=True).to(device, non_blocking=True)


                # Compute the logits and loss
                with torch.cuda.amp.autocast():
                    logits, token_logist, logits_2, _, _, triplet_loss = combiner((reference_feas, reference_middle_feas), (text_inputs, text_inputs_wo_neg), (target_feas, target_middle_feas))
                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    loss = crossentropy_criterion(logits, ground_truth)
                    loss_token = crossentropy_criterion(token_logist, ground_truth)
                    loss_2 = crossentropy_criterion(logits_2, ground_truth)

                    loss = loss + loss_token+ loss_2 + triplet_loss
                # Backpropagate and update the weights
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                experiment.log_metric('similar_per', logits_2.shape[1]-logits_2.shape[0], step=step)
                train_running_results['similar_per'] = (logits_2.shape[1]-logits_2.shape[0]) / batch_size
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
            clip_model = clip_model.float()  # In validation we use fp32 CLIP model
            with experiment.validate():
                with torch.no_grad():
                    combiner.eval()

                    # Compute and log validation metrics
                    results = compute_val_metrics(relative_val_dataset, clip_model, val_index_features,
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
                        if save_best and results_dict['arithmetic_mean'] > best_arithmetic:
                            best_arithmetic = results_dict['arithmetic_mean']
                            save_model('combiner_arithmetic', epoch, combiner, training_path)
                        if not save_best:
                            save_model(f'combiner_{epoch}', epoch, combiner, training_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='FineCVR', help="should be 'FineCVR'")
    parser.add_argument("--data_pth", type=str, default="FINECVR_PATH_ROOT",
                        help="the dataset's path")
    parser.add_argument("--dataset_op", type=str,
                        default="caption_by_CLIP__objects_scenes_attributes_threshold_0.17_version_11",
                        help="the dataset's option")
    parser.add_argument("--api-key", type=str, help="api for Comet logging")
    parser.add_argument("--workspace", type=str, help="workspace of Comet logging")
    parser.add_argument("--experiment-name", type=str, help="name of the experiment on Comet")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--num-epochs", default=30, type=int, help="number training epochs")
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
    parser.add_argument("--save_name", default='FDCA', action='store_true',
                        help="Save only the best model during training")

    args = parser.parse_args()
    if args.dataset.lower() not in ['FineCVR']:
        raise ValueError("Dataset should be 'FineCVR'")

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
