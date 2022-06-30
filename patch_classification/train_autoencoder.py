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

from dataset import RailSem19CroppedDatasetSegmentation, ImageNetDatasetSegmentation
from discriminator import Discriminator
from autoencoder_networks import AeSegParam02
from torchgeometry.losses.ssim import SSIM
from torch.autograd.variable import Variable
from histogram_loss import compute_histogram_loss


def get_transform(train, args):
    if train:
        # base_size is normal size of image, is then up/downsampled by 0.5 / 2, then image is cropped to crop_size
        return presets.SegmentationPresetTrain(base_size=224, crop_size=224)
    else:
        # image is rescaled to base_size
        return presets.SegmentationPresetEval(base_size=224, crop="ImageNet" in args.data_path)

def compute_gradient_image(image, device):
    sobel_x = torch.from_numpy(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])/8).float().unsqueeze(0).unsqueeze(0).to(device)
    sobel_y = torch.from_numpy(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])/8).float().unsqueeze(0).unsqueeze(0).to(device)
    #print(f"sobel_x: {sobel_x.shape}")
    G_x1 = torch.nn.functional.conv2d(image[:, 0, ::].unsqueeze(1), sobel_x, stride=1, padding=1)
    G_x2 = torch.nn.functional.conv2d(image[:, 1, ::].unsqueeze(1), sobel_x, stride=1, padding=1)
    G_x3 = torch.nn.functional.conv2d(image[:, 2, ::].unsqueeze(1), sobel_x, stride=1, padding=1)
    G_x = (G_x1 + G_x2 + G_x3) / 3
    #print(f"G_x1: {G_x1.shape}")
    #print(f"G_x: {G_x.shape}, min: {torch.min(G_x)}, max: {torch.max(G_x)}")
    G_y1 = torch.nn.functional.conv2d(image[:, 0, ::].unsqueeze(1), sobel_y, stride=1, padding=1)
    G_y2 = torch.nn.functional.conv2d(image[:, 1, ::].unsqueeze(1), sobel_y, stride=1, padding=1)
    G_y3 = torch.nn.functional.conv2d(image[:, 2, ::].unsqueeze(1), sobel_y, stride=1, padding=1)
    G_y = (G_y1 + G_y2 + G_y3) / 3
    #print(f"G_y1: {G_y1.shape}")
    # gradient_image = torch.square(G_x) + torch.square(G_y)
    gradient_image = torch.cat((G_x, G_y), dim=1)
    #print(f"Image: {image.shape}, min: {torch.min(image)}, max: {torch.max(image)}")
    #print(f"G_image: {gradient_image.shape}, min: {torch.min(gradient_image)}, max: {torch.max(gradient_image)}")
    return gradient_image


def criterion(args, inputs, target_seg, target_ae, ssim, model, model_d, criterion_d, device):
    losses = {}
    target_seg_detached = target_seg.detach()
    target_seg = torch.squeeze(target_seg, 1)
    losses["out_seg"] = nn.functional.cross_entropy(inputs["out_seg"], target_seg, ignore_index=255)
    losses["out_ae"] = 0
    if args.w_mse > 0:
        losses["out_mse"] = nn.functional.mse_loss(inputs["out_aa"], target_ae)
        losses["out_ae"] += args.w_mse * losses["out_mse"]
    if args.w_mae > 0:
        losses["out_mae"] = nn.functional.l1_loss(inputs["out_aa"], target_ae)
        losses["out_ae"] += args.w_mae * losses["out_mae"]
    if args.w_ssim > 0:
        losses["out_ssim"] = torch.mean(ssim(inputs["out_aa"], target_ae))
        losses["out_ae"] += args.w_ssim * losses["out_ssim"]
    if args.w_gan > 0:
        N = inputs["out_aa"].size(0)
        if args.cgan == 1:
            input_d = torch.cat((inputs["out_aa"], target_seg.unsqueeze(1)), dim=1)
        else:
            input_d = inputs["out_aa"]
        pred_discriminator = model_d(input_d)
        losses["out_generator"] = criterion_d(pred_discriminator, Variable(torch.ones(N, 1)).to(device))
        losses["out_ae"] += args.w_gan * losses["out_generator"]
    if args.w_emd > 0:
        emd, mi = compute_histogram_loss(inputs["out_aa"], target_ae, device, args.hist_cs, args.g_act)
        losses["out_emd"] = emd
        losses["out_ae"] += args.w_emd * losses["out_emd"]
        losses["out_mi"] = mi
        losses["out_ae"] += args.w_mi * losses["out_mi"]

    losses["combined"] = args.w_seg * losses["out_seg"] + losses["out_ae"]
    return losses


def evaluate(args, epoch, model, data_loader, model_d, criterion_d, device):
    model.eval()
    confmat = utils.ConfusionMatrix(2)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # Set up AE metric dict
    metric_dict = dict()
    if args.w_mse > 0:
        metric_dict["mse"] = 0
    if args.w_mae > 0:
        metric_dict["mae"] = 0
    if args.w_ssim > 0:
        metric_dict["ssim"] = 0
    if args.w_gan > 0:
        metric_dict["generator"] = 0
        metric_dict["discriminator"] = 0
    if args.w_emd > 0:
        metric_dict["emd"] = 0
        metric_dict["mi"] = 0
    metric_dict["ae_combined"] = 0

    ssim = SSIM(11)
    gaussian_blur = torchvision.transforms.GaussianBlur((29, 29), 20)

    with torch.no_grad():
        for idx, (image, target) in enumerate(metric_logger.log_every(data_loader, 100, header)):
            image, target_seg = image.to(device), target.to(device)
            target_ae = image.clone().detach()
            if args.g_act == "tanh":
                target_ae, _ = presets.denormalize_tanh(target_ae, target_ae)  # (-1, 1)
            else:
                target_ae, _ = presets.denormalize(target_ae, target_ae)  # (0, 1)

            ### Evaluate Discriminator
            if args.w_gan > 0:
                pred_ae = model(image)["out_aa"].detach()
                d_error, d_pred_real, d_pred_fake = evaluate_discriminator(args, model_d, criterion_d, target_ae, pred_ae, target_seg, device=device)
                metric_dict["discriminator"] += d_error

            outputs = model(image)
            # Segmentation
            output_seg = outputs["out_seg"]
            confmat.update(target_seg.flatten(), output_seg.argmax(1).flatten())
            # Autoencoder
            if args.optimize_with_mask > 0:
                target_seg_detached = target_seg.detach()
                optimization_mask = target_seg_detached.repeat((1, 3, 1, 1))
                outputs["out_aa"] = outputs["out_aa"] * optimization_mask
                target_ae = target_ae * optimization_mask

            # Visualization
            if idx in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90] and epoch == 0:
                if args.g_act == "tanh":
                    target_ae_pil, _ = presets.re_convert_tanh(target_ae.cpu(), target_ae.cpu())
                else:
                    target_ae_pil, _ = presets.re_convert(target_ae.cpu(), target_ae.cpu())
                target_ae_pil.save(os.path.join(args.output_dir, f"image{idx:02d}_val_gt.jpeg"))
            if idx in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90] and epoch % args.save_freq == 0:
                if args.g_act == "tanh":
                    pred_ae_pil, _ = presets.re_convert_tanh(outputs["out_aa"].cpu(), outputs["out_aa"].cpu())
                else:
                    pred_ae_pil, _ = presets.re_convert(outputs["out_aa"].cpu(), outputs["out_aa"].cpu())
                pred_ae_pil.save(os.path.join(args.output_dir, f"image{idx:02d}_val_epoch_{epoch:03d}.jpeg"))

            # Autoencoder
            if args.w_mse > 0:
                metric_mse = nn.functional.mse_loss(outputs["out_aa"], target_ae)
                metric_dict["mse"] += metric_mse
                metric_dict["ae_combined"] += args.w_mse * metric_mse
            if args.w_mae > 0:
                metric_mae = nn.functional.l1_loss(outputs["out_aa"], target_ae)
                metric_dict["mae"] += metric_mae
                metric_dict["ae_combined"] += args.w_mae * metric_mae
            if args.w_ssim > 0:
                metric_ssim = torch.mean(ssim(outputs["out_aa"], target_ae))
                metric_dict["ssim"] += metric_ssim
                metric_dict["ae_combined"] += args.w_ssim * metric_ssim
            if args.w_gan > 0:
                N = outputs["out_aa"].size(0)
                if args.cgan == 1:
                    input_d = torch.cat((outputs["out_aa"], target_seg), dim=1)
                else:
                    input_d = outputs["out_aa"]
                pred_discriminator = model_d(input_d)
                metric_generator = criterion_d(pred_discriminator, Variable(torch.ones(N, 1)).to(device))
                metric_dict["generator"] += metric_generator
                metric_dict["ae_combined"] += args.w_gan * metric_generator
            if args.w_emd > 0:
                emd, mi = compute_histogram_loss(outputs["out_aa"], target_ae, device, args.hist_cs, args.g_act)
                metric_dict["emd"] += emd
                metric_dict["ae_combined"] += args.w_emd * emd
                metric_dict["mi"] += mi
                metric_dict["ae_combined"] += args.w_mi * mi


        for metric_type in metric_dict.keys():
            metric_dict[metric_type] /= len(data_loader)
        confmat.reduce_from_all_processes()

    return confmat, metric_dict

def evaluate_discriminator(args, model_d, criterion_d, target_ae, pred_ae, target_seg, device):
    model_d.eval()
    with torch.no_grad():
        N = target_ae.size(0)
        # Evaluate on real data
        if args.cgan == 1:
            input_d_real = torch.cat((target_ae, target_seg), dim=1)
        else:
            input_d_real = target_ae
        prediction_real = model_d(input_d_real)
        error_real = args.w_gan * criterion_d(prediction_real, Variable(torch.ones(N, 1)).to(device))
        # Evaluate on fake data
        if args.cgan == 1:
            input_d_fake = torch.cat((pred_ae, target_seg), dim=1)
        else:
            input_d_fake = pred_ae
        prediction_fake = model_d(input_d_fake)
        error_fake = args.w_gan * criterion_d(prediction_fake, Variable(torch.zeros(N, 1)).to(device))
    return error_real + error_fake, prediction_real, prediction_fake

def train_discriminator(args, model_d, criterion_d, optimizer_d, target_ae, pred_ae, target_seg, device):
    model_d.train()
    optimizer_d.zero_grad()
    N = target_ae.size(0)
    # Train on real data
    if args.cgan == 1:
        input_d_real = torch.cat((target_ae, target_seg), dim=1)
    else:
        input_d_real = target_ae
    prediction_real = model_d(input_d_real)
    error_real = args.w_gan / 2 * criterion_d(prediction_real, Variable(args.label_smoothing * torch.ones(N, 1)).to(device))  # one-sided label smoothing
    error_real.backward()
    # Train on fake data
    if args.cgan == 1:
        input_d_fake = torch.cat((pred_ae, target_seg), dim=1)
    else:
        input_d_fake = pred_ae
    prediction_fake = model_d(input_d_fake)
    error_fake = args.w_gan / 2 * criterion_d(prediction_fake, Variable(torch.zeros(N, 1)).to(device))
    error_fake.backward()
    # Update weights
    optimizer_d.step()
    return error_real + error_fake, prediction_real, prediction_fake

def train_one_epoch(args, model, model_d, criterion, criterion_d, optimizer, optimizer_d, data_loader, device, epoch, print_freq,
                    counter, writer):
    model.train()
    model_d.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    loss_dict_epoch = None
    ssim = SSIM(11)
    gaussian_blur = torchvision.transforms.GaussianBlur((29, 29), 20)
    for idx, (image, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, target_seg = image.to(device), target.to(device)
        target_ae = image.clone().detach()
        if args.g_act == "tanh":
            target_ae, _ = presets.denormalize_tanh(target_ae, target_ae) # (-1, 1)
        else:
            target_ae, _ = presets.denormalize(target_ae, target_ae) # (0, 1)

        ### Train Discriminator
        if args.w_gan > 0:
            pred_ae = model(image)["out_aa"].detach()
            d_error, d_pred_real, d_pred_fake = train_discriminator(args, model_d, criterion_d, optimizer_d, target_ae, pred_ae, target_seg, device=device)

        ### Train Generator
        output = model(image)

        if args.optimize_with_mask > 0:
            target_seg_detached = target_seg.detach()
            optimization_mask = target_seg_detached.repeat((1, 3, 1, 1))
            output["out_aa"] = output["out_aa"] * optimization_mask
            target_ae = target_ae * optimization_mask

        # Visualization
        # if idx in [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90] and epoch in [128, 129, 130]: # epoch == 128:
        #     if args.g_act == "tanh":
        #         target_ae_pil, _ = presets.re_convert_tanh(target_ae[0, ::].cpu(), target_ae[0, ::].cpu())
        #     else:
        #         target_ae_pil, _ = presets.re_convert(target_ae[0, ::].cpu(), target_ae[0, ::].cpu())
        #     target_ae_pil.save(os.path.join(args.output_dir, f"image{idx:02d}_train_gt.jpeg"))
        # if idx in [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90] and epoch in [199, 200, 201]: # epoch % args.save_freq == 0:
        #     if args.g_act == "tanh":
        #         pred_ae_pil, _ = presets.re_convert_tanh(output["out_aa"][0, ::].cpu(), output["out_aa"][0, ::].cpu())
        #         target_ae_pil, _ = presets.re_convert_tanh(target_ae[0, ::].cpu(), target_ae[0, ::].cpu())
        #     else:
        #         pred_ae_pil, _ = presets.re_convert(output["out_aa"][0, ::].cpu(), output["out_aa"][0, ::].cpu())
        #         target_ae_pil, _ = presets.re_convert(target_ae[0, ::].cpu(), target_ae[0, ::].cpu())
        #     pred_ae_pil.save(os.path.join(args.output_dir, f"image{idx:02d}_train_epoch_{epoch:03d}.jpeg"))
        #     target_ae_pil.save(os.path.join(args.output_dir, f"image{idx:02d}_train_epoch_{epoch:03d}_gt.jpeg"))
        loss_dict = criterion(args, output, target_seg, target_ae, ssim, model, model_d, criterion_d, device=device)
        loss = loss_dict["combined"]
        # tensorboard
        if args.w_gan > 0:
            loss_dict["discriminator"] = d_error
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
    if "ImageNet" in args.data_path:
        dataset = ImageNetDatasetSegmentation(args.data_path, get_transform(train=True, args=args),
                                                      mode="train", train_fraction=args.train_fraction)
        dataset_test = ImageNetDatasetSegmentation(args.data_path, get_transform(train=False, args=args),
                                                           mode="val", train_fraction=args.train_fraction)
    else:
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


    if args.ae_type == "AeSegParam02_8110":
        model = AeSegParam02(c_seg=8, c_ae=1, c_param=1, mode="none", ratio=args.color_space_ratio, act=args.g_act)
    elif args.ae_type == "AeSegParam02_8210":
        model = AeSegParam02(c_seg=8, c_ae=2, c_param=1, mode="none", ratio=args.color_space_ratio, act=args.g_act)
    elif args.ae_type == "AeSegParam0_8410":
        model = AeSegParam02(c_seg=8, c_ae=4, c_param=1, mode="none", ratio=args.color_space_ratio, act=args.g_act)
    elif args.ae_type == "AeSegParam02_8810":
        model = AeSegParam02(c_seg=8, c_ae=8, c_param=1, mode="none", ratio=args.color_space_ratio, act=args.g_act)
    elif args.ae_type == "AeSegParam02_8111":
        model = AeSegParam02(c_seg=8, c_ae=1, c_param=1, mode="remap", ratio=args.color_space_ratio, act=args.g_act)
    elif args.ae_type == "AeSegParam02_8211":
        model = AeSegParam02(c_seg=8, c_ae=2, c_param=1, mode="remap", ratio=args.color_space_ratio, act=args.g_act)
    elif args.ae_type == "AeSegParam02_8411":
        model = AeSegParam02(c_seg=8, c_ae=4, c_param=1, mode="remap", ratio=args.color_space_ratio, act=args.g_act)
    elif args.ae_type == "AeSegParam02_8811":
        model = AeSegParam02(c_seg=8, c_ae=8, c_param=1, mode="remap", ratio=args.color_space_ratio, act=args.g_act)
    elif args.ae_type == "AeSegParam02_8112":
        model = AeSegParam02(c_seg=8, c_ae=1, c_param=1, mode="zero", ratio=args.color_space_ratio, act=args.g_act)
    elif args.ae_type == "AeSegParam02_8212":
        model = AeSegParam02(c_seg=8, c_ae=2, c_param=1, mode="zero", ratio=args.color_space_ratio, act=args.g_act)
    elif args.ae_type == "AeSegParam02_8412":
        model = AeSegParam02(c_seg=8, c_ae=4, c_param=1, mode="zero", ratio=args.color_space_ratio, act=args.g_act)
    elif args.ae_type == "AeSegParam02_8812":
        model = AeSegParam02(c_seg=8, c_ae=8, c_param=1, mode="zero", ratio=args.color_space_ratio, act=args.g_act)

    else:
        print(f"No such autoencoder type: {args.ae_type}!")
        return -1

    model.to(device)

    print(f"Model {args.ae_type} loaded.")

    if "AeSegParam" in args.ae_type:
        params_to_optimize = [
            {"params": [p for p in model.ae_decoder.parameters() if p.requires_grad]},
            {"params": [p for p in model.seg_decoder.parameters() if p.requires_grad]},
            {"params": [p for p in model.encoder.parameters() if p.requires_grad]},
            {"params": [p for p in model.param_network_1.parameters() if p.requires_grad]},
        ]

    if args.cgan == 1:
        model_d = Discriminator(input_channels=4)
    else:
        model_d = Discriminator(input_channels=3)
    model_d.to(device)
    params_to_optimize_d = [
        {"params": [p for p in model_d.discriminator.parameters() if p.requires_grad]}
    ]

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer_d = torch.optim.SGD(params_to_optimize_d, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, betas=(args.momentum, 0.999),
                                     weight_decay=args.weight_decay)
        optimizer_d = torch.optim.Adam(params_to_optimize_d, lr=args.lr_d, betas=(args.momentum, 0.999),
                                       weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=args.patience
    )
    criterion_d = nn.BCELoss()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=not args.test_only)
        model_d.load_state_dict(checkpoint["model_d"], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer_d.load_state_dict(checkpoint["optimizer_d"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    if args.test_only:
        confmat, metric_dict = evaluate(args, model, data_loader_test, model_d, criterion_d, device=device)
        print(confmat)
        print(f"Combined AE score: {metric_dict['ae_combined'].item()}")
        return

    # tensorboard
    writer = SummaryWriter(f"tensorboard_runs/{args.run_name}")
    counter = dict()
    counter["batch"] = 0
    counter["epoch"] = 0

    start_time = time.time()

    print("Starting training ... ")
    current_best_epoch = 0
    current_best_score = 9999
    for epoch in range(args.start_epoch, args.epochs):
        counter["epoch"] = epoch
        train_one_epoch(args, model, model_d, criterion, criterion_d, optimizer, optimizer_d, data_loader, device, epoch,
                        args.print_freq, counter, writer)
        confmat, metric_dict = evaluate(args, epoch, model, data_loader_test, model_d, criterion_d, device=device)

        # update learning rate
        if metric_dict["ae_combined"].item() < current_best_score:
            current_best_score = metric_dict["ae_combined"].item()
            current_best_epoch = epoch
        lr_scheduler.step(current_best_score)
        print(f"Current best: Epoch {current_best_epoch}: {current_best_score}")

        # tensorboard
        acc_global, acc, iou = confmat.compute()
        writer.add_scalar(f"Validation/acc_global", acc_global.item(), counter["epoch"])
        for idx, a in enumerate(acc):
            writer.add_scalar(f"Validation/acc_{idx}", a.item(), counter["epoch"])
        for idx, i in enumerate(iou):
            writer.add_scalar(f"Validation/iou_{idx}", i.item(), counter["epoch"])
        for metric_type in metric_dict.keys():
            writer.add_scalar(f"Validation/{metric_type}", metric_dict[metric_type].item(), counter["epoch"])
        print(confmat)
        print(f"Combined AE score: {metric_dict['ae_combined'].item()}")

        checkpoint = {
            "model": model.state_dict(),
            "model_d": model_d.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
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

    parser.add_argument("--data_path", default="./datasets/Railsem19Croppedv1.h5", type=str, help="dataset path")
    parser.add_argument("--seed", default=0, type=str, help="Seed for random generator")
    parser.add_argument("--train_fraction", default=0.9, type=float, help="fraction of train images")
    parser.add_argument("--run_name", default="ae", type=str, help="name of training run")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch_size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to run")

    parser.add_argument(
        "-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers (default: 0)"
    )
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--lr_d", default=0.01, type=float, help="initial learning rate of discriminator")
    parser.add_argument("--ae_type", default="AeSegParam02_8810", type=str, help="which Autoencoder")
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
    parser.add_argument("--w_seg", default=0.0, type=float, help="segmentation weight")
    parser.add_argument("--w_mse", default=0.0, type=float, help="mse weight")
    parser.add_argument("--w_mae", default=0.0, type=float, help="mae weight")
    parser.add_argument("--w_ssim", default=0.0, type=float, help="ssim weight")
    parser.add_argument("--w_gan", default=0.0, type=float, help="gan weight")
    parser.add_argument("--w_emd", default=0.0, type=float, help="emd histogram loss weight")
    parser.add_argument("--w_mi", default=0.0, type=float, help="mi histogram loss weight")
    parser.add_argument("--cgan", default=1, type=int, help="whether to make discriminator conditional")
    parser.add_argument("--g_act", default="tanh", type=str, help="generator activation")
    parser.add_argument("--hist_cs", default="yuv", type=str, help="histogram color space")
    parser.add_argument("--label_smoothing", default=0.9, type=float, help="label smoothing parameter, should be close to 1")
    parser.add_argument("--color_space_ratio", default=0.1, type=float, help="color space ratio for each channel, NOT relevant for our experiments")
    parser.add_argument("--output_dir", default="./trained_models", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer: sgd or adam")
    parser.add_argument("--patience", default=200, type=int, help="number of epochs after which to reduce learning rate by 1/10")
    parser.add_argument("--optimize_with_mask", default=0, type=int, help="whether to apply segmentation mask to the networks output before optimization")
    parser.add_argument(
        "--test_only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )


    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
