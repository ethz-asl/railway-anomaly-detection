import datetime
import os
import time

import presets
import torch
import torch.utils.data
import torchvision
import utils
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json
import numpy as np

from dataset import RailSem19CroppedDatasetSegmentation
from patchclass_networks import PatchSegModelLight
from torchvision.transforms import functional as F
import random


def get_transform(train, args):
    if train:
        # base_size is normal size of image, is then up/downsampled by 0.5 / 2, then image is cropped to crop_size
        return presets.SegmentationPresetTrain(base_size=224, crop_size=224)
    else:
        # image is rescaled to base_size
        return presets.SegmentationPresetEval(base_size=224)


def criterion(args, outputs_student, normalized_teacher):
    losses = {}
    losses["out_student_mse"] = nn.functional.mse_loss(outputs_student, normalized_teacher)
    losses["combined"] = losses["out_student_mse"]
    return losses


def evaluate(args, model_student, model_teacher, data_loader, device, mean, std):
    model_student.eval()
    model_teacher.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    score = 0
    img_score_vector = torch.zeros(len(data_loader))
    with torch.no_grad():
        for idx, (image, target) in enumerate(metric_logger.log_every(data_loader, 100, header)):
            image = image.to(device)
            target_seg = target.to(device)
            outputs_teacher = model_teacher(image)["descriptor"]
            normalized_teacher = F.normalize(outputs_teacher, mean=mean, std=std).clone().detach()
            outputs_student = model_student(image)["descriptor"]
            if args.optimize_with_mask > 0:
                _, C, _, _ = outputs_student.shape
                target_seg_detached = target_seg.detach()
                optimization_mask = target_seg_detached.repeat((1, C, 1, 1))
                outputs_student = outputs_student * optimization_mask
                normalized_teacher = normalized_teacher * optimization_mask
            #diff_raw = torch.squeeze(torch.square(outputs_student - normalized_teacher))
            #diff_raw = torch.mean(diff_raw, dim=0)
            score += nn.functional.mse_loss(outputs_student, normalized_teacher)
            img_score_vector[idx] = nn.functional.mse_loss(outputs_student, normalized_teacher)
        score = score / len(data_loader)
        score_std, score_mean = torch.std_mean(img_score_vector)

    return score, score_std, score_mean


def train_one_epoch(args, model_student, model_teacher, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq,
                    counter, writer, mean, std):
    model_student.train()
    model_teacher.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    loss_dict_epoch = None
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image = image.to(device)
        target_seg = target.to(device)
        outputs_teacher = model_teacher(image)["descriptor"]
        normalized_teacher = F.normalize(outputs_teacher, mean=mean, std=std).clone().detach()
        # normalized_teacher = outputs_teacher.sub_(mean).div_(std).clone().detach()
        outputs_student = model_student(image)["descriptor"]
        # diff_raw = torch.squeeze(torch.square(outputs_student - normalized_teacher))
        # diff_raw = torch.mean(diff_raw, dim=0)
        if args.optimize_with_mask > 0:
            _, C, _, _ = outputs_student.shape
            target_seg_detached = target_seg.detach()
            optimization_mask = target_seg_detached.repeat((1, C, 1, 1))
            outputs_student = outputs_student * optimization_mask
            normalized_teacher = normalized_teacher * optimization_mask

        loss_dict = criterion(args, outputs_student, normalized_teacher)
        print(f"Output student: {outputs_student.shape}")
        print(f"Normalized teacher: {outputs_student.shape}")
        print(f"Loss: {loss_dict['out_student_mse']}")
        loss = loss_dict["combined"]

        # tensorboard
        for loss_type in loss_dict.keys():
            writer.add_scalar(f"LossBatch/{loss_type}", loss_dict[loss_type].item(), counter["batch"])
        counter["batch"] += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # tensorboard
        if loss_dict_epoch is None:
            loss_dict_epoch = dict()
            for loss_type in loss_dict.keys():
                loss_dict_epoch[loss_type] = loss_dict[loss_type]
        else:
            for loss_type in loss_dict.keys():
                loss_dict_epoch[loss_type] = loss_dict_epoch[loss_type] + loss_dict[loss_type]

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

    # tensorboard
    for loss_type in loss_dict_epoch.keys():
        loss_dict_epoch[loss_type] /= len(data_loader)
        writer.add_scalar(f"LossEpoch/{loss_type}", loss_dict_epoch[loss_type].item(), counter["epoch"])


def main(args):
    # Random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_flush_denormal(True)
    # Logging
    date_time = datetime.now()
    date_time_string = date_time.strftime("%Y%m%d_%H%M%S")
    args.run_name = f"{args.run_name}_{date_time_string}"
    utils.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    utils.mkdir(args.output_dir)
    args.mean_std_dir = os.path.dirname(args.teacher_checkpoint)
    args.mean_std_suffix = os.path.basename(args.teacher_checkpoint[:-4])
    # Log args
    args_path = os.path.join(args.output_dir, "args.txt")
    with open(args_path, 'w') as file:
        file.write(json.dumps(vars(args)))

    print(args)

    device = torch.device(args.device)
    print(f"Running on device: {device}")

    # our dataset has two classes only - background and person
    num_classes = 21
    # use our dataset and defined transformations
    dataset = RailSem19CroppedDatasetSegmentation(args.data_path, get_transform(train=True, args=args),
                                                  mode="train", train_fraction=args.train_fraction)
    dataset_test = RailSem19CroppedDatasetSegmentation(args.data_path, get_transform(train=False, args=args),
                                                       mode="val", train_fraction=args.train_fraction)
    print("Datasets loaded.")

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    print("Dataloaders created.")


    if args.model == "patchsegmodellight":
        model_teacher = PatchSegModelLight(in_channels=3, out_channels=128, stages=args.stages, patch_only=False)
        model_student = PatchSegModelLight(in_channels=3, out_channels=128, stages=args.stages, patch_only=False)
    else:
        print(f"Model {args.model} undefined!")
        return

    checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu")
    model_student.load_state_dict(checkpoint["model"], strict=True)
    model_teacher.load_state_dict(checkpoint["model"], strict=True)

    model_teacher.to(device)
    model_student.to(device)

    with open(os.path.join(args.mean_std_dir, f"{args.mean_std_suffix}_mean.npy"), "rb") as file:
        mean = np.load(file)
    with open(os.path.join(args.mean_std_dir, f"{args.mean_std_suffix}_std.npy"), "rb") as file:
        std = np.load(file)

    mean = torch.from_numpy(mean).to(device)
    print(f"Mean shape: {mean.shape}")
    std = torch.from_numpy(std).to(device)
    print(f"Std shape: {std.shape}")

    print("Model loaded.")

    if args.model == "patchsegmodellight":
        params_to_optimize = [
            {"params": [p for p in model_student.block_1.parameters() if p.requires_grad]},
        ]

    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, betas=(args.momentum, 0.999),
                                     weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_student.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1

    # tensorboard
    writer = SummaryWriter(f"tensorboard_runs/{args.run_name}")
    counter = dict()
    counter["batch"] = 0
    counter["epoch"] = 0

    start_time = time.time()

    print("Starting training ... ")
    current_best_epoch = 0
    current_best_score = 99999
    for epoch in range(args.start_epoch, args.epochs):
        counter["epoch"] = epoch
        train_one_epoch(args, model_student, model_teacher, criterion, optimizer, data_loader, lr_scheduler, device, epoch,
                        args.print_freq, counter, writer, mean=mean, std=std)
        score, score_std, score_mean = evaluate(args, model_student, model_teacher, data_loader_test, device=device, mean=mean, std=std)

        # tensorboard
        writer.add_scalar(f"Validation/feature_mse_std", score_std.item(), counter["epoch"])
        writer.add_scalar(f"Validation/feature_mse_mean", score_mean.item(), counter["epoch"])
        writer.add_scalar(f"Validation/feature_mse", score.item(), counter["epoch"])
        print(f"student_mse score: {score.item()}")
        print(f"student_mse mean: {score_mean.item()}")
        print(f"student_mse std: {score_std.item()}")

        # update learning rate
        if score.item() < current_best_score:
            current_best_score = score.item()
            current_best_epoch = epoch
        lr_scheduler.step(current_best_score)
        print(f"Current best: Epoch {current_best_epoch}: {current_best_score}")

        checkpoint = {
            "model": model_student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if epoch % args.save_freq == 0:
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    writer.flush()


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--data_path", default="./datasets/Railsem19Croppedv1.h5", type=str, help="dataset path")
    parser.add_argument("--train_fraction", default=0.9, type=float, help="fraction of train images")
    parser.add_argument("--model", default="patchsegmodellight", type=str, help="model name")
    parser.add_argument("--run_name", default="student", type=str, help="name of training run")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch_size", default=1, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")

    parser.add_argument(
        "-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers (default: 0)"
    )
    parser.add_argument("--lr", default=0.0001, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight_decay",
        default=1e-5,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-5)",
        dest="weight_decay",
    )
    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
    parser.add_argument("--stages", default=2, type=int, help="number of stages of neural network, stages=1 corresponds to patch_size of 17, stages=2 to 33, stage=3 to 65")
    parser.add_argument("--output_dir", default="./trained_models", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--teacher_checkpoint", default="", type=str, help="path of teacher checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--save_freq", default=1, type=int, help="save frequency")
    parser.add_argument("--optimize_with_mask", default=1, type=int, help="whether to apply segmentation mask to the networks output before optimization")
    parser.add_argument("--seed", default=0, type=int, help="Seed for random generator")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
