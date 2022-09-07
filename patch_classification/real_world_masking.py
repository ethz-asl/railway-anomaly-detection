import os
import numpy as np

import presets
import torch
import torch.utils.data
import torchvision
import utils
from torch import nn
from PIL import Image
from PIL import ImageDraw

from dataset import RealWorldDatasetRaw
from autoencoder_networks import AeSegParam02
from torchgeometry.losses.ssim import SSIM
from patchclass_networks import PatchClassModel, PatchSegModelLight
from torchvision.transforms import functional as F


CONFIG = dict()
CONFIG["deeplabv3"] = {"model_name": "deeplabv3_resnet50",
     "checkpoint_name": "./trained_models/deeplabv3_model_5.pth",
     "ae_model_name": "none",
     "ae_checkpoint_name": "none",
     "stages": 99,
     "g_act": "tanh",
     "patch_size": 51,
     "column1": "DeeplabV3\n",
     "column2": "-",
    }

def evaluate(args, ae_model, ae_model_name, model, model_name, data_loader, g_act, obstacle_threshold, patch_size, device,
             vis_path=None, mean=None, std=None, teacher_model=None, student1_model=None, student2_model=None, student3_model=None):
    if model_name == "students":
        teacher_model.eval()
        student1_model.eval()
        student2_model.eval()
        student3_model.eval()
    else:
        if ae_model:
            ae_model.eval()
        if model:
            model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    if vis_path:
        utils.mkdir(vis_path)
    header = "Test:"

    stat_book = dict()
    stat_book["image_log"] = list()
    stat_book["images_fp"] = list()
    stat_book["images_fn_correct"] = list()
    stat_book["conf_correct"] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    storage = list()

    # Compute max if necessary:
    if model_name == "mse":
        overall_max = 0.986
    elif model_name == "students":
        overall_max = 5
    else:
        overall_max = 1.0

    with torch.no_grad():
        for idx, (image, name, size) in enumerate(
                metric_logger.log_every(data_loader, 100, header)):

            name = name[0] # Quick and dirty fix (tuple comes from some transformations
            print(f"Image {name} ...")

            image = image.to(device)


            # Visualize original image
            if g_act == "tanh":
                image_target_ae, _ = presets.denormalize_tanh(image, image)  # (-1, 1)
                image_vis_pil, _ = presets.re_convert_tanh(image_target_ae, image_target_ae)
            else:
                image_target_ae, _ = presets.denormalize(image, image)  # (0, 1)
                image_vis_pil, _ = presets.re_convert(image_target_ae, image_target_ae)
            image_vis = np.asarray(image_vis_pil)

            size= ( int(size[0]), int(size[1]) ) # quick and dirty fix, some conversion happenend


            #image_ae_vis_pil = Image.fromarray(image_ae_vis)
            #image_ae_vis_pil.save(os.path.join(args.output_path, f"{args.config}_ae_{name}.png"))

            # Prepare input for PatchSeg model
            input_seg = image

            # Inference
            with torch.no_grad():
                output_seg = model(input_seg)["out"]
                output_seg = nn.functional.softmax(output_seg, dim=1)
                output_seg = output_seg[0, 0, ::]

            mask = output_seg.clone().to(device)
            mask[output_seg > obstacle_threshold] = 0
            mask[output_seg <= obstacle_threshold] = 1

            if args.visualize > 0:
                # Visualized output seg masked
                mask_vis_gray = 1-output_seg
                mask_vis_gray = mask_vis_gray * 255
                mask_vis_gray = presets.torch_mask_to_pil(mask_vis_gray)
                mask_vis = Image.new("RGB", mask_vis_gray.size)
                mask_vis.paste(mask_vis_gray)
                mask_vis = np.asarray(mask_vis)
                mask_vis_pil = Image.fromarray(mask_vis)
                mask_vis_pil = mask_vis_pil.resize(size)
                mask_vis_pil.save(os.path.join(args.output_path, f"{name}_vis1mask.png"))

                mask_vis_gray = mask
                mask_vis_gray = mask_vis_gray * 255
                mask_vis_gray = presets.torch_mask_to_pil(mask_vis_gray)
                mask_vis = Image.new("RGB", mask_vis_gray.size)
                mask_vis.paste(mask_vis_gray)
                mask_vis = np.asarray(mask_vis)
                mask_vis_pil = Image.fromarray(mask_vis)
                mask_vis_pil = mask_vis_pil.resize(size)
                mask_vis_pil.save(os.path.join(args.output_path, f"{name}_vis2mask.png"))

            # Visualized output seg masked
            mask_pil = presets.torch_mask_to_pil(mask)
            mask_pil = mask_pil.resize(size) # Resize!
            mask_pil.save(os.path.join(args.output_path, f"{name}_mask.png"))

    return storage

def get_model(model_name, checkpoint_name, ae_model_name, ae_checkpoint_name, stages, g_act, device):

    print(f"Running on device: {device}")

    # Gan model
    if ae_model_name == "AeSegParam02_8810":
        ae_model = AeSegParam02(c_seg=8, c_ae=8, c_param=1, mode="none", ratio=0, act=g_act)
    else:
        ae_model = None
        print(f"No autoencoder used!")

    if ae_model:
        ae_model.to(device)
        ae_checkpoint = torch.load(ae_checkpoint_name, map_location="cpu")
        ae_model.load_state_dict(ae_checkpoint["model"], strict=False)
        print("AE Model loaded.")
    mean = None
    std = None

    # Segmentation model
    if model_name == "deeplabv3_resnet50":
        model = torchvision.models.segmentation.__dict__[model_name](
                pretrained=False,
                pretrained_backbone=False,
                num_classes=2,
                aux_loss=False,
            )
    elif model_name == "patchdiff":
        model = PatchClassModel(stages=stages, in_channels=6)
    elif model_name == "patchclass":
        model = PatchClassModel(stages=stages, in_channels=3)

    else:
        model = None
        print("No seg model!")

    if model:
        model.to(device)
        checkpoint = torch.load(checkpoint_name, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)

    return model, ae_model, mean, std

def main(args):
    device = torch.device("cpu")
    # Create Dataset
    dataset_test = RealWorldDatasetRaw(args.data_path)
    print("Dataset loaded.")
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=1
    )
    print("Dataloader created.")

    method = CONFIG[args.config]
    # Get models:
    model_name = method["model_name"]

    print(f"Method {method['column1']} ... ")
    checkpoint_name = method["checkpoint_name"]
    ae_model_name = method["ae_model_name"]
    ae_checkpoint_name = method["ae_checkpoint_name"]
    stages = method["stages"]
    g_act = method["g_act"]
    threshold = args.threshold
    patch_size = method["patch_size"]
    model, ae_model, mean, std = get_model(model_name, checkpoint_name, ae_model_name, ae_checkpoint_name,
                                                   stages, g_act, device)
    method_data = evaluate(args, ae_model, ae_model_name, model, model_name, data_loader_test, g_act, threshold,
                                   patch_size, device, vis_path=args.output_path, mean=mean, std=std)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--data_path", default="./real_world_dataset_raw", type=str, help="dataset path")
    parser.add_argument("--output_path", default="./real_world_dataset_raw", type=str, help="output directory")
    parser.add_argument("--config", default="deeplabv3", choices=["deeplabv3"], type=str, help="which config/model to evaluate")
    parser.add_argument("--threshold", default=0.9, type=float,
                        help="which threshold to use for masking, the higher, the more railway!")
    parser.add_argument("--visualize", default=0, type=int,
                        help="whether or not to save visualization masks")
    parser.add_argument("--device", default="cpu", type=str, help="device (Use cuda or cpu)")

    return parser



if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)