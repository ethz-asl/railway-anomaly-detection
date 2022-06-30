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

from dataset import RailSem19CroppedDatasetPatchSegmentation 
from patchclass_networks import PatchClassModel, PatchSegModelLight


def criterion(args, inputs, target_seg, optimization_mask, weights, device):
    losses = {}
    if args.optimize_with_mask > 0:
        target_seg[optimization_mask == 0] = 255

    target_seg = torch.squeeze(target_seg, 1)
    losses["out_seg"] = nn.functional.cross_entropy(inputs["out"], target_seg, weight=weights, ignore_index=255)
    return losses


def evaluate(args, epoch, model, data_loader, device):
    model.eval()
    confmat = utils.ConfusionMatrix(2)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    with torch.no_grad():
        for idx, (image, target, optimization_mask) in enumerate(metric_logger.log_every(data_loader, 100, header)):
            image, target_seg = image.to(device), target.to(device)
            outputs = model(image)
            # Segmentation
            output_seg = outputs["out"]
            if args.test_only:
                img_pil, _ = presets.re_transform(image[0, 0:3, ::].cpu(), image[0, 0:3, ::].cpu())
                img_gan_pil, _ = presets.re_transform(image[0, 3:6, ::].cpu(), image[0, 3:6, ::].cpu())
                target_seg_pil = target_seg[0, 0, ::].cpu().numpy()
                target_seg_pil = np.uint8(target_seg_pil * 255)
                target_seg_pil = Image.fromarray(target_seg_pil, mode="L")
                img_pil.save(os.path.join(args.output_dir, f"image{idx:02d}_val_img.jpeg"))
                img_gan_pil.save(os.path.join(args.output_dir, f"image{idx:02d}_val_gan.jpeg"))
                target_seg_pil.save(os.path.join(args.output_dir, f"image{idx:02d}_val_target.jpeg"))
                output_seg = nn.functional.softmax(output_seg, dim=1)
                output_seg_pil = output_seg[0, 1, ::].cpu().numpy()
                output_seg_pil = np.uint8(output_seg_pil * 255)
                output_seg_pil = Image.fromarray(output_seg_pil, mode="L")
                output_seg_pil.save(os.path.join(args.output_dir, f"image{idx:02d}_val_pred.jpeg"))
            confmat.update(target_seg.flatten(), output_seg.argmax(1).flatten())
        confmat.reduce_from_all_processes()

    return confmat

def train_one_epoch(args, model, criterion, optimizer, data_loader, device, epoch, print_freq,
                    counter, writer):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    loss_dict_epoch = None

    # Weighted cross-entropy
    if args.ce_weight_1 == 0.5:
        ce_weights = None
    else:
        ce_weights = torch.Tensor([1-args.ce_weight_1, args.ce_weight_1]).to(device)

    for idx, (image, target, optimization_mask) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, target_seg, optimization_mask = image.to(device), target.to(device), optimization_mask.to(device)
        output = model(image)

        if idx in [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90] and False: # epoch % args.save_freq == 0:
            for i in range(10):
                img_pil, _ = presets.re_transform(image[0, 3*i:3*i+3, ::].cpu(), image[0, 0:3, ::].cpu())
                img_pil.save(os.path.join(args.output_dir, f"image{idx:02d}_{i:02d}_train_epoch_{epoch:03d}_img.jpeg"))
            target_seg_pil = target_seg[0, 0, ::].cpu().numpy()
            target_seg_pil = np.uint8(target_seg_pil*255)
            target_seg_pil = Image.fromarray(target_seg_pil, mode="L")
            target_seg_pil.save(os.path.join(args.output_dir, f"image{idx:02d}_train_epoch_{epoch:03d}_target.jpeg"))

        loss_dict = criterion(args, output, target_seg, optimization_mask, ce_weights, device)
        loss = loss_dict["out_seg"]
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
    # Log args
    args_path = os.path.join(args.output_dir, "args.txt")
    with open(args_path, 'w') as file:
        file.write(json.dumps(vars(args)))

    print(args)

    device = torch.device(args.device)
    print(f"Running on device: {device}")

    # use our dataset and defined transformations
    if args.dataset == "patch":
        dataset = RailSem19CroppedDatasetPatchSegmentation(args.data_path, args.data_path_in, mode="train", train_fraction=args.train_fraction, use_gan=args.use_gan > 0, use_neighbors=args.use_neighbors > 0, imagenet_ratio=args.imagenet_ratio, secimage_ratio=args.secimage_ratio, no_seg=args.no_seg>0)
        dataset_test = RailSem19CroppedDatasetPatchSegmentation(args.data_path, args.data_path_in, mode="val", train_fraction=args.train_fraction, use_gan=args.use_gan > 0, use_neighbors=args.use_neighbors > 0, imagenet_ratio=args.imagenet_ratio, secimage_ratio=args.secimage_ratio, no_seg=args.no_seg>0)
    else:
        print(f"Dataset {args.dataset} not available!")
        return -1
    print("Datasets loaded.")

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn_2,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn_2
    )

    print("Dataloaders created.")

    if args.model == "deeplabv3_resnet50":
        model = torchvision.models.segmentation.__dict__[args.model](
                pretrained=False,
                pretrained_backbone=args.pretrained_backbone>0,
                num_classes=2,
                aux_loss=False,
            )
        # model.backbone.conv1 = nn.Conv2d(dataset.channels, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
    elif args.model == "patchclassmodel":
        model = PatchClassModel(stages=args.stages, in_channels=dataset.channels)
    elif args.model == "patchsegmodellight":
        model = PatchSegModelLight(in_channels=dataset.channels, stages=args.stages, patch_only=False)

    model.to(device)

    print(f"Model {args.model} loaded.")

    if args.model == "patchclassmodel":
        params_to_optimize = [
            {"params": [p for p in model.block_1.parameters() if p.requires_grad]},
            {"params": [p for p in model.t_block_1.parameters() if p.requires_grad]},
            {"params": [p for p in model.block_2.parameters() if p.requires_grad]},
            {"params": [p for p in model.t_block_2.parameters() if p.requires_grad]},
            {"params": [p for p in model.classifier.parameters() if p.requires_grad]},
        ]
        if args.stages >= 3:
            params_to_optimize.append({"params": [p for p in model.block_3.parameters() if p.requires_grad]})
            params_to_optimize.append({"params": [p for p in model.t_block_3.parameters() if p.requires_grad]})
    elif args.model == "patchsegmodellight":
        params_to_optimize = [
            {"params": [p for p in model.block_1.parameters() if p.requires_grad]},
            {"params": [p for p in model.classifier.parameters() if p.requires_grad]},
        ]
    else:
        params_to_optimize = [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model.classifier.parameters() if p.requires_grad]},
        ]

    optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 20, 25], gamma=0.1
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    if args.test_only:
        confmat = evaluate(args, 0, model, data_loader_test, device=device)
        print(confmat)
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
        train_one_epoch(args, model, criterion, optimizer, data_loader, device, epoch,
                        args.print_freq, counter, writer)
        confmat = evaluate(args, epoch, model, data_loader_test, device=device)

        # tensorboard
        acc_global, acc, iou = confmat.compute()
        writer.add_scalar(f"Validation/acc_global", acc_global.item(), counter["epoch"])
        for idx, a in enumerate(acc):
            writer.add_scalar(f"Validation/acc_{idx}", a.item(), counter["epoch"])
        for idx, i in enumerate(iou):
            writer.add_scalar(f"Validation/iou_{idx}", i.item(), counter["epoch"])
        print(confmat)

        # update learning rate
        lr_scheduler.step()

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
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

    parser.add_argument("--data_path", default="./datasets/Railsem19CroppedGan017v1.h5", type=str, help="dataset path")
    parser.add_argument("--data_path_in", default="./datasets/ImageNet.h5", type=str, help="dataset path")
    parser.add_argument("--seed", default=0, type=str, help="Seed for random generator")
    parser.add_argument("--model", default="patchclassmodel", type=str, help="deeplabv3_resnet50 or patchclassmodel")
    parser.add_argument("--train_fraction", default=0.9, type=float, help="fraction of train images")
    parser.add_argument("--run_name", default="patchclass", type=str, help="name of training run")
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
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
    parser.add_argument("--save_freq", default=10, type=int, help="save frequency")
    parser.add_argument("--output_dir", default="./trained_models", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--test_only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--pretrained_backbone", default=1, type=int, help="use pretrained backbone for deeplabv3?")
    parser.add_argument("--flip", default=0.5, type=float, help="flip probability for original image during data loading")
    parser.add_argument("--crop", default=0.8, type=float, help="crop probability for original image during data loading")
    parser.add_argument("--stages", default=1, type=int, help="Number of stages of the Patch Classification network. Stage 0 corresponds to patch size 13, 1 to 21, 2 to 29, 3 to 35, and 4 to 51.")
    parser.add_argument("--dataset", default="patch", type=str, help="which dataset to use")
    parser.add_argument("--optimize_with_mask", default=0, type=int, help="whether to apply segmentation mask to the networks output before optimization")
    parser.add_argument("--ce_weight_1", default=0.5, type=float, help="cross entropy weight of class 1, class 0 will get 1 - that")
    parser.add_argument("--use_gan", default=1, type=int, help="whether to use gan for patch segmentation")
    parser.add_argument("--use_neighbors", default=0, type=int, help="whether to use neighboring patches in patch segmentation")
    parser.add_argument("--imagenet_ratio", default=0.5, type=float, help="ratio of imagenet images in patch segmentation")
    parser.add_argument("--secimage_ratio", default=0.0, type=float, help="ratio of different (second) images in patch segmentation")
    parser.add_argument("--no_seg", default=0, type=int, help="whether to not use segmentation")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
