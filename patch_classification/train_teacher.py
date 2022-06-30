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
import random
import numpy as np
from PIL import Image

from dataset import ImageNetDatasetPatch
from patchclass_networks import PatchSegModelLight


def compactness_loss(output, normalize=False):
    # dim: (batch, vector, 1, 1)
    output = output.squeeze(-1)
    output = output.squeeze(-1)
    _, n = output.size()
    if normalize:
        avg = torch.mean(output, axis=0)
        std = torch.std(output, axis=0)
        output_normalized = output - avg
        output_normalized /= std
    else:
        output_normalized = output
    corr = torch.matmul(output_normalized, output_normalized.T) / n
    loss = torch.sum(torch.triu(corr, diagonal=1))
    return loss

def criterion(args, output_orig, output_plus, output_minus, device):
    losses = {}
    losses["out_triplet"] = nn.functional.triplet_margin_loss(output_orig, output_plus, output_minus, swap=True)
    losses["out_compact"] = compactness_loss(output_orig)
    losses["combined"] = losses["out_triplet"] + losses["out_compact"]
    return losses


def evaluate(args, epoch, model, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    metric_dict = dict()
    metric_dict["triplet"] = 0
    metric_dict["compact"] = 0
    metric_dict["combined"] = 0

    with torch.no_grad():
        for idx, (image_orig, image_plus, image_minus) in enumerate(metric_logger.log_every(data_loader, 100, header)):
            image_orig, image_plus, image_minus = image_orig.to(device), image_plus.to(device), image_minus.to(device)
            output_orig = model(image_orig)["out"]
            output_plus = model(image_plus)["out"]
            output_minus = model(image_minus)["out"]
            loss_dict = criterion(args, output_orig, output_plus, output_minus, device)
            loss_triplet = loss_dict["out_triplet"]
            metric_dict["triplet"] += loss_triplet
            loss_compact = loss_dict["out_compact"]
            metric_dict["compact"] += loss_compact / (args.batch_size * args.batch_size)
            loss_combined = loss_dict["combined"]
            metric_dict["combined"] += loss_combined
            if idx >= 6399:
                break
    for metric_type in metric_dict.keys():
        metric_dict[metric_type] /= 6400

    return metric_dict

def train_one_epoch(args, model, criterion, optimizer, data_loader, device, epoch, print_freq,
                    counter, writer):
    # Random seed
    seed = args.seed + epoch
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_flush_denormal(True)

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    loss_dict_epoch = None

    for idx, (image_orig, image_plus, image_minus) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image_orig, image_plus, image_minus = image_orig.to(device), image_plus.to(device), image_minus.to(device)
        output_orig = model(image_orig)["out"]
        output_plus = model(image_plus)["out"]
        output_minus = model(image_minus)["out"]

        if False: # for debug visualizations
            img_orig_pil, _ = presets.re_transform(image_orig[0, ::].cpu(), image_orig[0, ::].cpu())
            img_orig_pil.save(os.path.join(args.output_dir, f"image{idx:02d}_orig_train_epoch_{epoch:03d}.jpeg"))
            img_plus_pil, _ = presets.re_transform(image_plus[0, ::].cpu(), image_plus[0, ::].cpu())
            img_plus_pil.save(os.path.join(args.output_dir, f"image{idx:02d}_plus_train_epoch_{epoch:03d}.jpeg"))
            img_minus_pil, _ = presets.re_transform(image_minus[0, ::].cpu(), image_minus[0, ::].cpu())
            img_minus_pil.save(os.path.join(args.output_dir, f"image{idx:02d}_minus_train_epoch_{epoch:03d}.jpeg"))

        loss_dict = criterion(args, output_orig, output_plus, output_minus, device)
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

        # One epoch is defined as 1000 iterations!
        if idx >= 999:
            break

    # tensorboard
    for loss_type in loss_dict_epoch.keys():
        loss_dict_epoch[loss_type] /= 1000
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
    # Log args
    args_path = os.path.join(args.output_dir, "args.txt")
    with open(args_path, 'w') as file:
        file.write(json.dumps(vars(args)))

    print(args)

    device = torch.device(args.device)
    print(f"Running on device: {device}")

    # dataset
    dataset = ImageNetDatasetPatch(args.data_path, mode="train", train_fraction=args.train_fraction, patch_size=args.patch_size)
    dataset_test = ImageNetDatasetPatch(args.data_path, mode="val", train_fraction=args.train_fraction, patch_size=args.patch_size)


    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers
    )

    print("Dataloaders created.")

    if args.model == "patchsegmodellight":
        model = PatchSegModelLight(in_channels=3, out_channels=128, stages=args.stages, patch_only=True)

    model.to(device)

    print(f"Model {args.model} loaded.")

    if args.model == "patchsegmodellight":
        params_to_optimize = [
            {"params": [p for p in model.block_1.parameters() if p.requires_grad]},
        ]
    else:
        print("Undefined model!")

    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, betas=(args.momentum, 0.999),
                                     weight_decay=args.weight_decay)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1

    if args.test_only:
        triplet_loss_val = evaluate(args, 0, model, data_loader_test, device=device)["triplet"]
        print(f"Triplet loss val: {triplet_loss_val}")
        return

    # tensorboard
    writer = SummaryWriter(f"tensorboard_runs/{args.run_name}")
    counter = dict()
    counter["batch"] = 0
    counter["epoch"] = 0

    start_time = time.time()

    print("Starting training ... ")
    for epoch in range(args.start_epoch, args.epochs):
        counter["epoch"] = epoch
        train_one_epoch(args, model, criterion, optimizer, data_loader, device, epoch, args.print_freq, counter, writer)
        combined_loss_val = evaluate(args, epoch, model, data_loader_test, device=device)["combined"]

        # tensorboard
        writer.add_scalar(f"Validation/triplet_loss", combined_loss_val.item(), counter["epoch"])

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if epoch % args.save_freq == 0:
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    writer.flush()


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--data_path", default="./datasets/ImageNet.h5", type=str, help="dataset path")
    parser.add_argument("--seed", default=0, type=int, help="Seed for random generator")
    parser.add_argument("--model", default="patchsegmodellight", type=str, help="patchsegmodellight")
    parser.add_argument("--train_fraction", default=0.9, type=float, help="fraction of train images")
    parser.add_argument("--run_name", default="teacher", type=str, help="name of training run")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch_size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to run")

    parser.add_argument(
        "-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers (default: 0)"
    )
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
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
    parser.add_argument("--save_freq", default=1, type=int, help="save frequency")
    parser.add_argument("--output_dir", default="./trained_models", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--test_only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--stages", default=2, type=int, help="number of stages of neural network, stages=1 corresponds to patch_size of 17, stages=2 to 33, stage=3 to 65")
    parser.add_argument("--patch_size", default=32, type=int, help="student patch size for training")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
