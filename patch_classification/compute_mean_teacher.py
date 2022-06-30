import datetime
import os

import presets
import torch
import torch.utils.data
import utils
from datetime import datetime
from dataset import RailSem19CroppedDatasetSegmentation
from patchclass_networks import PatchSegModelLight
import numpy as np


def get_transform(train, args):
    if train:
        # base_size is normal size of image, is then up/downsampled by 0.5 / 2, then image is cropped to crop_size
        return presets.SegmentationPresetTrain(base_size=224, crop_size=224)
    else:
        # image is rescaled to base_size
        return presets.SegmentationPresetEval(base_size=224)


def compute_mean(args, model_teacher, data_loader, device):
    model_teacher.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Mean:"
    mean_acc = 0
    with torch.no_grad():
        for idx, (image, target) in enumerate(metric_logger.log_every(data_loader, 100, header)):
            print(f"Mean: {idx}/{len(data_loader)}")
            image = image.to(device)
            outputs_teacher = model_teacher(image)["descriptor"]
            outputs_teacher = torch.squeeze(outputs_teacher)
            mean_img = torch.mean(outputs_teacher, dim=(1, 2))
            #diff_raw = torch.squeeze(torch.square(outputs_student - normalized_teacher))
            #diff_raw = torch.mean(diff_raw, dim=0)
            mean_acc += mean_img.cpu()
        print(f"Dataloader length: {len(data_loader)}")
        mean = mean_acc / len(data_loader)
        print(f"Mean Total: {mean.shape}")
    return mean

def compute_std(args, model_teacher, data_loader, device, mean):
    model_teacher.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Std:"
    std_acc = 0
    mean_repeated = mean.repeat((224, 224, 1)).to(device)
    mean_repeated = torch.permute(mean_repeated, (2, 0, 1))
    print(f"Repeated mean shape: {mean_repeated.shape}")
    with torch.no_grad():
        for idx, (image, target) in enumerate(metric_logger.log_every(data_loader, 100, header)):
            print(f"Std: {idx}/{len(data_loader)}")
            image = image.to(device)
            outputs_teacher = model_teacher(image)["descriptor"]
            outputs_teacher = torch.squeeze(outputs_teacher)
            outputs_size = outputs_teacher.shape[1] * outputs_teacher.shape[2]
            std_acc_img = torch.sum(torch.square(outputs_teacher-mean_repeated), dim=(1, 2))
            #diff_raw = torch.squeeze(torch.square(outputs_student - normalized_teacher))
            #diff_raw = torch.mean(diff_raw, dim=0)
            std_acc += std_acc_img.cpu()
        print(f"Dataloader length: {len(data_loader)}, image size: {outputs_size}")
        std = std_acc / (len(data_loader) * outputs_size)
        std = torch.sqrt(std)
        print(f"Std Total: {std.shape}")
    return std


def main(args):

    t = torch.tensor([1, 2, 3])
    t_repeated = t.repeat((224,224,1))
    t_repeated = torch.permute(t_repeated, (2, 0, 1))
    # print(t_repeated.shape)
    # print(t_repeated[0,:,:])
    # Logging
    date_time = datetime.now()
    date_time_string = date_time.strftime("%Y%m%d_%H%M%S")
    # Log args
    args.output_dir = os.path.dirname(args.teacher_checkpoint)
    args.mean_std_suffix = os.path.basename(args.teacher_checkpoint[:-4])
    # args_path = os.path.join(args.output_dir, "args_compute_mean.txt")
    # with open(args_path, 'w') as file:
    #     file.write(json.dumps(vars(args)))
    print(args)

    device = torch.device(args.device)
    print(f"Running on device: {device}")

    # use our dataset and defined transformations
    dataset = RailSem19CroppedDatasetSegmentation(args.data_path, get_transform(train=True, args=args),
                                                  mode="train", train_fraction=args.train_fraction)
    print("Datasets loaded.")

    train_sampler = torch.utils.data.RandomSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=train_sampler,
        num_workers=1,
        collate_fn=utils.collate_fn,
        drop_last=True,
    )
    print("Dataloaders created.")

    if args.model == "patchsegmodellight":
        model_teacher = PatchSegModelLight(in_channels=3, out_channels=512, stages=args.stages, patch_only=False)
    else:
        print(f"Model {args.model} undefined!")
        return

    checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu")
    model_teacher.load_state_dict(checkpoint["model"], strict=True)
    model_teacher.to(device)
    print("Model loaded.")


    print("Starting training ... ")
    mean = compute_mean(args, model_teacher, data_loader, device)
    std = compute_std(args, model_teacher, data_loader, device, mean)
    mean = mean.numpy()
    std = std.numpy()
    print(f"Mean: {mean.shape}")
    print(mean)
    print(f"Std: {std.shape}")
    print(std)
    with open(os.path.join(args.output_dir, f"{args.mean_std_suffix}_mean.npy"), "wb") as file:
        np.save(file, mean)
    with open(os.path.join(args.output_dir, f"{args.mean_std_suffix}_std.npy"), "wb") as file:
        np.save(file, std)



def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--data_path", default="./datasets/Railsem19Croppedv1.h5", type=str, help="dataset path")
    parser.add_argument("--train_fraction", default=0.9, type=float, help="fraction of train images")
    parser.add_argument("--model", default="patchsegmodellight", type=str, help="model name")
    parser.add_argument("--teacher_checkpoint", default="", type=str, help="checkpoint for teacher network")
    parser.add_argument("--stages", default=1, type=int, help="number of stages for patchsegmodellight")
    parser.add_argument("--device", default="cpu", type=str, help="device (Use cuda or cpu Default: cuda)")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
