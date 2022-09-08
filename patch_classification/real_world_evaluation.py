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

from dataset import RealWorldDataset, RealWorldDataset2
from autoencoder_networks import AeSegParam02
from torchgeometry.losses.ssim import SSIM
from sklearn.metrics import roc_curve, auc
from patchclass_networks import PatchClassModel, PatchSegModelLight
from torchvision.transforms import functional as F


CONFIG = dict()
CONFIG["deeplabv3"] = {"model_name": "deeplabv3_resnet50",
     "checkpoint_name": "./trained_models/deeplabv3_model_5.pth",
     "ae_model_name": "none",
     "ae_checkpoint_name": "none",
     "stages": 99,
     "g_act": "tanh",
     "threshold": 0.3,
     "patch_size": 51,
     "column1": "DeeplabV3\n",
     "column2": "-",
    }

CONFIG["ae_rmse"] = {"model_name": "mse",
     "checkpoint_name": "none",
     "ae_model_name": "AeSegParam02_8810",
     "ae_checkpoint_name": "./trained_models/ae_mse_model_199.pth",
     "stages": 99,
     "g_act": "tanh",
     "threshold": 0.2,
     "patch_size": 7,
     "column1": "RMSE AE\n"r"$\mathcal{L}_{MSE}$",
     "column2": r"$\mathcal{L}_{MSE}$",
    }

CONFIG["ae_ssim"] = {"model_name": "ssim",
     "checkpoint_name": "none",
     "ae_model_name": "AeSegParam02_8810",
     "ae_checkpoint_name": "./trained_models/ae_ssim_model_199.pth",
     "stages": 99,
     "g_act": "tanh",
     "threshold": 0.65,
     "patch_size": 21,
     "column1": "SSIM AE\n"r"$\mathcal{L}_{SSIM}$",
     "column2": r"$\mathcal{L}_{SSIM}$",
    }

CONFIG["students"] = {"model_name": "students",
     "teacher_checkpoint_name": "./trained_models/teacher_33_model_49.pth",
     "student1_checkpoint_name": "./trained_models/t33_student_1_model_40.pth",
     "student2_checkpoint_name": "./trained_models/t33_student_2_model_40.pth",
     "student3_checkpoint_name": "./trained_models/t33_student_3_model_40.pth",
     "stages": 2,
     "g_act": "tanh",
     "threshold": 0.2,
     "patch_size": 35,
     "column1": "Students33\n",
    }

CONFIG["patchclass"] = {"model_name": "patchclass",
     "checkpoint_name": "./trained_models/patchclass_21_model_20.pth",
     "ae_model_name": "none",
     "ae_checkpoint_name": "none",
     "stages": 1,
     "g_act": "tanh",
     "threshold": 0.95,
     "patch_size": 11,
     "column1": "PatchClass21\n",
     "column2": "-",
    }

CONFIG["patchdiff_mse"] = {"model_name": "patchdiff",
     "checkpoint_name": "./trained_models/patchdiff_21_mse_model_25.pth",
     "ae_model_name": "AeSegParam02_8810",
     "ae_checkpoint_name": "./trained_models/ae_mse_model_199.pth",
     "stages": 1,
     "g_act": "tanh",
     "threshold": 0.95,
     "patch_size": 7,
     "column1": "PatchDiff21\n"r"$\mathcal{L}_{MSE}$",
     "column2": r"$\mathcal{L}_{MSE}$",
    }

CONFIG["patchdiff_ssim"] = {"model_name": "patchdiff",
     "checkpoint_name": "./trained_models/patchdiff_21_ssim_model_20.pth",
     "ae_model_name": "AeSegParam02_8810",
     "ae_checkpoint_name": "./trained_models/ae_ssim_model_199.pth",
     "stages": 1,
     "g_act": "tanh",
     "threshold": 0.95,
     "patch_size": 7,
     "column1": "PatchDiff21\n"r"$\mathcal{L}_{SSIM}$",
     "column2": r"$\mathcal{L}_{SSIM}$",
    }

CONFIG["patchdiff_gan"] = {"model_name": "patchdiff",
     "checkpoint_name": "./trained_models/patchdiff_21_gan_model_30.pth",
     "ae_model_name": "AeSegParam02_8810",
     "ae_checkpoint_name": "./trained_models/ae_gan_model_199.pth",
     "stages": 1,
     "g_act": "tanh",
     "threshold": 0.95,
     "patch_size": 11,
     "column1": "PatchDiff21\n"r"$\mathcal{L}_{GAN}$",
     "column2": r"$\mathcal{L}_{GAN}$"
    }

CONFIG["patchdiff_gan+hist"] = {"model_name": "patchdiff",
     "checkpoint_name": "./trained_models/patchdiff_21_gan+hist_model_20.pth",
     "ae_model_name": "AeSegParam02_8810",
     "ae_checkpoint_name": "./trained_models/ae_gan+hist_model_199.pth",
     "stages": 1,
     "g_act": "tanh",
     "threshold": 0.95,
     "patch_size": 7,
     "column1": "PatchDiff21\n"r"$\mathcal{L}_{GAN} + \mathcal{L}_{HIST}$",
     "column2": r"$\mathcal{L}_{GAN} + \mathcal{L}_{HIST}$",
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
    stat_book["auroc_data"] = np.zeros((1001, 2))

    storage = list()

    # Compute max if necessary:
    if model_name == "mse":
        overall_max = 0.986
    elif model_name == "students":
        overall_max = 5
    else:
        overall_max = 1.0

    with torch.no_grad():
        for idx, (image, target, annotation, name) in enumerate(
                metric_logger.log_every(data_loader, 100, header)):

            name = name[0] # Quick and dirty fix (tuple comes from some transformations
            print(f"Image {name} ...")
            idx_results = dict()

            image, target_seg_orig = image.to(device), target.to(device)
            #target_seg_orig = target_seg.clone().to(device)

            # Mask for evaluation (discard background)
            evaluation_mask = target_seg_orig == 1

            if args.obstacle_segmentation > 0:
                target_seg = annotation.to(device)
                target_seg[torch.logical_not(evaluation_mask)] = 0
                target_seg_masked = target_seg.clone().type(torch.FloatTensor)

            evaluation_mask = evaluation_mask.squeeze()

            # Visualize original image
            if g_act == "tanh":
                image_target_ae, _ = presets.denormalize_tanh(image, image)  # (-1, 1)
                image_vis_pil, _ = presets.re_convert_tanh(image_target_ae, image_target_ae)
            else:
                image_target_ae, _ = presets.denormalize(image, image)  # (0, 1)
                image_vis_pil, _ = presets.re_convert(image_target_ae, image_target_ae)
            image_vis = np.asarray(image_vis_pil)
            idx_results[f"image"] = image_vis

            if ae_model:
                # Run  AE inference
                with torch.no_grad():
                    outputs = ae_model(image)
                output_ae = outputs["out_aa"]

                # Post-process AE image for PatchSeg
                if g_act == "tanh":
                    image_ae = (output_ae / 2) + 0.5
                else:
                    image_ae = output_ae
                image_ae = torchvision.transforms.functional.normalize(image_ae, mean=(0.485, 0.456, 0.406),
                                                                           std=(0.229, 0.224, 0.225))

                # Visualize AE image
                if g_act == "tanh":
                    image_ae_vis, _ = presets.re_convert_tanh(output_ae, output_ae)  # no de-normalization
                else:
                    image_ae_vis, _ = presets.re_convert(output_ae, output_ae)  # no de-normalization

                image_ae_vis = np.asarray(image_ae_vis)
            else:
                image_ae_vis = np.ones_like(image_vis) * 255  # dummy
            idx_results[f"ae"] = image_ae_vis
            image_ae_vis_pil = Image.fromarray(image_ae_vis)
            image_ae_vis_pil.save(os.path.join(args.output_path, f"{args.config}_ae_{name}.png"))

            # Prepare input for PatchSeg model
            if model_name == "patchdiff":
                input_seg = torch.cat((image, image_ae), dim=1)
            else:
                input_seg = image

            # Inference
            if model_name == "patchclass" or model_name == "patchdiff":
                with torch.no_grad():
                    output_seg = model(input_seg)["out"]
                    output_seg = nn.functional.softmax(output_seg, dim=1)
                    output_seg = output_seg[0, 0, ::]
            elif model_name == "deeplabv3_resnet50":
                with torch.no_grad():
                    output_seg = model(input_seg)["out"]
                    output_seg = nn.functional.softmax(output_seg, dim=1)
                    output_seg = output_seg[0, 0, ::]
            elif model_name == "mse":
                if g_act == "tanh":
                    image_target_ae = (image_target_ae / 2) + 0.5
                    output_ae = (output_ae / 2) + 0.5
                output_seg = torch.squeeze(torch.sqrt(torch.square(image_target_ae - output_ae)))
                output_seg = torch.mean(output_seg, dim=0)
            elif model_name == "ssim":
                ssim = SSIM(11)
                if g_act == "tanh":
                    image_target_ae = (image_target_ae / 2) + 0.5
                    output_ae = (output_ae / 2) + 0.5
                output_seg = torch.squeeze(ssim(image_target_ae, output_ae)) * 2  # SSIM output is in range (0, 0.5)
                output_seg = torch.mean(output_seg, dim=0)
            elif model_name == "students":
                with torch.no_grad():
                    outputs_teacher = teacher_model(input_seg)["descriptor"]
                    normalized_teacher = F.normalize(outputs_teacher, mean=mean["teacher"],
                                                     std=std["teacher"]).clone().detach()
                    outputs_student1 = student1_model(input_seg)["descriptor"]
                    outputs_student2 = student2_model(input_seg)["descriptor"]
                    outputs_student3 = student3_model(input_seg)["descriptor"]
                    output_e_students = 1 / 3 * (outputs_student1 + outputs_student2 + outputs_student3)
                    output_e = torch.squeeze(torch.square(normalized_teacher - output_e_students))
                    output_e = torch.sum(output_e, dim=0)
                    output_e_normalized = torch.abs((output_e - mean["e"]) / std["e"])
                    output_v_mean = torch.sum(torch.squeeze(torch.square(output_e_students)), dim=0)
                    output_v_student1 = torch.sum(torch.squeeze(torch.square(outputs_student1)), dim=0)
                    output_v_student2 = torch.sum(torch.squeeze(torch.square(outputs_student2)), dim=0)
                    output_v_student3 = torch.sum(torch.squeeze(torch.square(outputs_student3)), dim=0)
                    output_v = 1 / 3 * (output_v_student1 + output_v_student2 + output_v_student3) - output_v_mean
                    output_v_normalized = torch.abs((output_v - mean["v"]) / std["v"])
                    output_seg = output_e_normalized + output_v_normalized

            # Make sure segmentation outputs are in range (0, 1):
            output_seg[output_seg > overall_max] = overall_max
            output_seg = output_seg / overall_max
            # Check if segmentation outputs are in range (0, 1):
            if torch.max(output_seg) > 1 or torch.min(output_seg) < 0:
                print("ERROR: Output segmentation out of range!")
                return

            # Compute whether an obstacle can be found in seg output based on patch density:
            kernel = torch.tensor(np.ones((patch_size, patch_size)) * 1 / (patch_size * patch_size)).view(1, 1,
                                                                                                              patch_size,
                                                                                                              patch_size).type(
                    torch.FloatTensor)
            output_seg_masked = output_seg.clone()
            output_seg_masked[torch.logical_not(evaluation_mask)] = 0
            patch_density = torch.nn.functional.conv2d(output_seg_masked.unsqueeze(0).unsqueeze(1), kernel,
                                                           padding='same')
            max_patch_density = torch.max(patch_density)
            patch_density[patch_density <= obstacle_threshold] = 0
            patch_density.squeeze()
            if max_patch_density > obstacle_threshold:
                found_obstacle = 1
                # Compute centroid based on patch density
                x = torch.linspace(0, 223, steps=224).unsqueeze(0)
                x = x.repeat(224, 1)
                y = torch.linspace(0, 223, steps=224).unsqueeze(1)
                y = y.repeat(1, 224)
                centroid_x = int(
                        torch.floor(torch.sum(patch_density * x) / torch.sum(patch_density)).type(torch.LongTensor))
                centroid_y = int(
                        torch.floor(torch.sum(patch_density * y) / torch.sum(patch_density)).type(torch.LongTensor))
            else:
                found_obstacle = 0

            # Compute whether an obstacle can be found in groundtruth:
            if args.obstacle_segmentation > 0:
                if torch.max(target_seg_masked) == 1:
                    has_obstacle = 1
                    # get bounding box
                    target_seg_masked_pil = presets.torch_mask_to_pil(target_seg_masked)
                    l_bb, u_bb, r_bb, d_bb = target_seg_masked_pil.getbbox()
                else:
                    has_obstacle = 0
                # Binned data for AUROC
                if has_obstacle:
                    for i in range(224):
                        for j in range(224):
                            if evaluation_mask[i, j]:
                                val = int(output_seg[i, j] * 1000)
                                if target_seg_masked[0, 0, i, j] == 1:
                                    stat_book["auroc_data"][val, 1] += 1
                                else:
                                    stat_book["auroc_data"][val, 0] += 1
            else:
                has_obstacle = annotation[0][0]
                l_bb, u_bb, r_bb, d_bb = annotation[0][1], annotation[0][2], annotation[0][3], annotation[0][4]

            # Check whether found obstacle was correct
            if found_obstacle == 1 and has_obstacle == 1 and l_bb < centroid_x < r_bb and u_bb < centroid_y < d_bb:
                found_correct = True
            else:
                found_correct = False

            # Now fill statistics book
            image_data = {"idx": idx, "has_obstacle": has_obstacle, "found_obstacle": found_obstacle,
                              "found_correct": found_correct}
            stat_book["image_log"].append(image_data)

            # Fill confusion matrices:
            if has_obstacle == 1:
                if found_correct == 1:
                    stat_book["conf_correct"]["tp"] += 1
                else:
                    stat_book["conf_correct"]["fn"] += 1
                    stat_book["images_fn_correct"].append(idx)
            else:  # if there is no obstacle, it does not matter if it was classified correctly or not
                if found_obstacle == 1:
                    stat_book["conf_correct"]["fp"] += 1
                    stat_book["images_fp"].append(idx)
                else:
                    stat_book["conf_correct"]["tn"] += 1

            # Visualize Patch Density
            patch_density[patch_density > 0] = 1
            patch_density_vis_gray = patch_density * 255
            patch_density_vis_gray = presets.torch_mask_to_pil(patch_density_vis_gray)
            patch_density_vis = Image.new("RGB", patch_density_vis_gray.size)
            patch_density_vis.paste(patch_density_vis_gray)
            patch_density_vis = Image.blend(patch_density_vis, image_vis_pil, 0.5)
            draw = ImageDraw.Draw(patch_density_vis)
            if found_obstacle == 1:
                if found_correct == 1:
                    draw.ellipse((centroid_x - 5, centroid_y - 5, centroid_x + 5, centroid_y + 5), fill="green")
                else:
                    draw.ellipse((centroid_x - 5, centroid_y - 5, centroid_x + 5, centroid_y + 5), fill="red")
            if has_obstacle == 1:
                if found_correct == 1:
                    draw.rectangle((l_bb, u_bb, r_bb, d_bb), outline="green", width=5)
                else:
                    draw.rectangle((l_bb, u_bb, r_bb, d_bb), outline="red", width=5)
            patch_density_vis = np.asarray(patch_density_vis)
            idx_results[f"pred_detect"] = patch_density_vis
            patch_density_vis_pil = Image.fromarray(patch_density_vis)
            patch_density_vis_pil.save(os.path.join(args.output_path, f"{args.config}_loc_{name}.png"))

            # Visualized output seg masked
            output_seg_masked_vis_gray = output_seg
            output_seg_masked_vis_gray[torch.logical_not(evaluation_mask)] = 0.5
            output_seg_masked_vis_gray = output_seg_masked_vis_gray * 255
            output_seg_masked_vis_gray = presets.torch_mask_to_pil(output_seg_masked_vis_gray)
            output_seg_masked_vis = Image.new("RGB", output_seg_masked_vis_gray.size)
            output_seg_masked_vis.paste(output_seg_masked_vis_gray)
            draw = ImageDraw.Draw(output_seg_masked_vis)
            output_seg_masked_vis = np.asarray(output_seg_masked_vis)
            idx_results[f"pred_seg"] = output_seg_masked_vis
            output_seg_masked_vis_pil = Image.fromarray(output_seg_masked_vis)
            output_seg_masked_vis_pil.save(os.path.join(args.output_path, f"{args.config}_seg_{name}.png"))

            storage.append(idx_results)

        # Compute global metrics:
        if args.obstacle_segmentation > 0:
            fpr_list, tpr_list, thr_list, auc = roc_curve(stat_book["auroc_data"])
            roc_auc = auc(fpr_list, tpr_list)

        tp = stat_book["conf_correct"]["tp"]
        fp = stat_book["conf_correct"]["fp"]
        tn = stat_book["conf_correct"]["tn"]
        fn = stat_book["conf_correct"]["fn"]
        tot = tp + fp + tn + fn
        pp = tp + fp
        pn = tn + fn
        p = tp + fn
        n = tn + fp
        if p != 0:
            tpr = tp / (
                        tp + fn)  # True Positive Rate (Sensitivity, Recall): ratio of (correct) detection if there was an obstacle (SAFETY-CRITICAL)
        else:
            tpr = 99999
        if pn != 0:
            npv = tn / (
                        fn + tn)  # Negative Predictive Value: ratio of correct non-detections if there was no detection (SAFETY-CRITICAL)
        else:
            npv = 99999
        if n != 0:
            tnr = tn / (
                        fp + tn)  # True Negative Rate (Specificity, 1-FPR): ratio of (correct) non-detection if there was no obstacle
        else:
            tnr = 99999
        if pp != 0:
            ppv = tp / (
                        tp + fp)  # Positive Predictive Value (Precision): ratio of correct detection if there was a detection
        else:
            ppv = 99999
        if tpr + ppv != 0:
            f1 = 2 * (tpr * ppv) / (tpr + ppv)  # F1 score
        else:
            f1 = -1
        output_string = ""
        if args.obstacle_segmentation > 0:
            output_string += f"AUROC: {roc_auc:.3f}\n\n"
        else:
            output_string += f"AUROC not available (no obstacle mask)\n\n"
        output_string += f"CONFMAT METRICS:\n\n"
        output_string += f"             \t Obstacle     No Obstacle\n"
        output_string += f"Detection:   \t  {tp:4d}          {fp:4d}           |   {pp:4d}\n"
        output_string += f"No Detection:\t  {fn:4d}          {tn:4d}           |   {pn:4d}\n"
        output_string += f"             \t----------------------------------------------\n"
        output_string += f"             \t  {p:4d}          {n:4d}           |   {tot:4d} images\n"
        output_string += f"\n"
        output_string += f"True Positive Rate (Sensitivity, Recall):\t {tpr:.3f} \t\t ratio of (correct) detection if there was an obstacle (SAFETY-CRITICAL)\n"
        output_string += f"Negative Predictive Value:               \t {npv:.3f} \t\t ratio of correct non-detections if there was no detection (SAFETY-CRITICAL)\n"
        output_string += f"True Negative Rate (Specificity, 1-FPR): \t {tnr:.3f} \t\t ratio of (correct) non-detection if there was no obstacle\n"
        output_string += f"Positive Predictive Value (Precision):   \t {ppv:.3f} \t\t ratio of correct detection if there was a detection\n"
        output_string += f"F1 Score:                                \t {f1:.3f} \t\t trade-off between precision and recall\n"
        output_string += f"\n\n"
        output_string += f"Image Ids for False Positive: {stat_book['images_fp']}\n"
        output_string += f"Image Ids for False Negative: {stat_book['images_fn_correct']}\n"
        output_string += f"\n\n"

        print(output_string)
        with open(os.path.join(args.output_path, f"{args.config}_f1.txt"), "w") as file:
            file.write(output_string)

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

def get_model_students(teacher_checkpoint_name, student1_checkpoint_name, student2_checkpoint_name, student3_checkpoint_name, stages, device):

    print(f"Running on device: {device}")

    # Models
    teacher = PatchSegModelLight(in_channels=3, out_channels=128, stages=stages, patch_only=False).to(device)
    student1 = PatchSegModelLight(in_channels=3, out_channels=128, stages=stages, patch_only=False).to(device)
    student2 = PatchSegModelLight(in_channels=3, out_channels=128, stages=stages, patch_only=False).to(device)
    student3 = PatchSegModelLight(in_channels=3, out_channels=128, stages=stages, patch_only=False).to(device)

    teacher_checkpoint = torch.load(teacher_checkpoint_name, map_location="cpu")
    teacher.load_state_dict(teacher_checkpoint["model"], strict=False)
    student1_checkpoint = torch.load(student1_checkpoint_name, map_location="cpu")
    student1.load_state_dict(student1_checkpoint["model"], strict=False)
    student2_checkpoint = torch.load(student2_checkpoint_name, map_location="cpu")
    student2.load_state_dict(student2_checkpoint["model"], strict=False)
    student3_checkpoint = torch.load(student3_checkpoint_name, map_location="cpu")
    student3.load_state_dict(student3_checkpoint["model"], strict=False)

    # Load Mean and Std (pre-computed)
    mean_std_dir = os.path.dirname(teacher_checkpoint_name)
    mean_std_suffix = os.path.basename(teacher_checkpoint_name[:-4])
    with open(os.path.join(mean_std_dir, f"{mean_std_suffix}_mean.npy"), "rb") as file:
        mean = np.load(file)
    with open(os.path.join(mean_std_dir, f"{mean_std_suffix}_std.npy"), "rb") as file:
        std = np.load(file)
    mean_teacher = torch.from_numpy(mean).to(device)
    # print(f"Mean shape: {mean.shape}")
    std_teacher = torch.from_numpy(std).to(device)
    # print(f"Std shape: {std.shape}")

    return teacher, student1, student2, student3, mean_teacher, std_teacher

def roc_curve(roc_data):
    max_thres = roc_data.shape[0] # 1001 in our case (threshold is from >= 0 (all) to >= 1001 (none)
    fpr_list = np.empty((max_thres+1,))
    tpr_list = np.empty((max_thres+1,))
    thr_list = np.empty((max_thres+1,))
    # >= 1001:
    fpr_list[max_thres] = 0
    tpr_list[max_thres] = 0
    thr_list[max_thres] = max_thres
    # i in [1000, 1]
    for i in range(max_thres - 1, 0, -1):
        thr_list[i] = i
        tp = np.sum(roc_data[i:, 1])
        fp = np.sum(roc_data[i:, 0])
        tn = np.sum(roc_data[:i, 0])
        fn = np.sum(roc_data[:i, 1])
        tpr = float(tp / (tp + fn))
        fpr = float(fp / (fp + tn))
        tpr_list[i] = tpr
        fpr_list[i] = fpr
    # >= 0:
    fpr_list[0] = 1
    tpr_list[0] = 1
    thr_list[0] = 0

    return fpr_list, tpr_list, thr_list, auc

def main(args):
    device = torch.device("cpu")
    # Create Dataset
    if args.obstacle_segmentation > 0:
        dataset_test = RealWorldDataset2(args.data_path)
    else:
        dataset_test = RealWorldDataset(args.data_path)
    print("Dataset loaded.")
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=1
    )
    print("Dataloader created.")

    method = CONFIG[args.config]
    # Get models:
    model_name = method["model_name"]
    if model_name == "students":
        teacher_checkpoint_name = method["teacher_checkpoint_name"]
        student1_checkpoint_name = method["student1_checkpoint_name"]
        student2_checkpoint_name = method["student2_checkpoint_name"]
        student3_checkpoint_name = method["student3_checkpoint_name"]
        stages = method["stages"]
        g_act = method["g_act"]
        threshold = method["threshold"]
        patch_size = method["patch_size"]
        teacher, student1, student2, student3, mean_teacher, std_teacher = get_model_students(
                teacher_checkpoint_name, student1_checkpoint_name, student2_checkpoint_name, student3_checkpoint_name,
                stages, device)
        mean = dict()
        mean["teacher"] = mean_teacher
        mean["v"] = 0.56520
        mean["e"] = 2.69930
        std = dict()
        std["teacher"] = std_teacher
        std["v"] = 1.20560
        std["e"] = 5.07015
        method_data = evaluate(args, None, "none", None, model_name, data_loader_test, g_act, threshold,
                                            patch_size, device, vis_path=args.output_path, mean=mean, std=std,
                               teacher_model=teacher, student1_model=student1, student2_model=student2, student3_model=student3)

    else:
        print(f"Method {method['column1']} ... ")
        checkpoint_name = method["checkpoint_name"]
        ae_model_name = method["ae_model_name"]
        ae_checkpoint_name = method["ae_checkpoint_name"]
        stages = method["stages"]
        g_act = method["g_act"]
        threshold = method["threshold"]
        patch_size = method["patch_size"]
        model, ae_model, mean, std = get_model(model_name, checkpoint_name, ae_model_name, ae_checkpoint_name,
                                                   stages, g_act, device)
        method_data = evaluate(args, ae_model, ae_model_name, model, model_name, data_loader_test, g_act, threshold,
                                   patch_size, device, vis_path=args.output_path, mean=mean, std=std)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--data_path", default="./real_world_dataset", type=str, help="dataset path")
    parser.add_argument("--output_path", default="./real_world_results", type=str, help="output directory")
    parser.add_argument("--obstacle_segmentation", default=1, type=int, help="whether to use existing obstacle segmentation <name>_obstacle.png (0 or 1)")
    parser.add_argument("--config", default="patchclass", choices=["deeplabv3", "ae_rmse", "ae_ssim", "patchclass",
                                                                   "patchdiff_mse", "patchdiff_ssim", "patchdiff_gan",
                                                                   "patchdiff_gan+hist", "students"], type=str, help="which config/model to evaluate")
    parser.add_argument("--device", default="cpu", type=str, help="device (Use cuda or cpu)")

    return parser



if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)