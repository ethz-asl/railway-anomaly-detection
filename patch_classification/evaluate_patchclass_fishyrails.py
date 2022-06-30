import os
import numpy as np

import presets
import torch
import torch.utils.data
import torchvision
import utils
from torch import nn
import json
from PIL import Image
from PIL import ImageDraw

from dataset import FishyrailsCroppedDataset, RailSem19CroppedDatasetLikeFishyrails
from autoencoder_networks import AeSegParam02
from torchgeometry.losses.ssim import SSIM
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from patchclass_networks import PatchClassModel, PatchSegModelLight
from torchvision.transforms import functional as F


def evaluate(args, ae_model, model, data_loader, device, num_classes, vis_path=None, mean=None, std=None):
    if ae_model:
        ae_model.eval()
    if model:
        model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    if vis_path:
        utils.mkdir(vis_path)
    header = "Test:"

    stat_book = dict()
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    stat_book["image_log"] = list()
    stat_book["images_fp"] = list()
    stat_book["images_fn_obstacle"] = list()
    stat_book["images_fn_correct"] = list()
    stat_book["conf_obstacle"] = list()
    stat_book["conf_correct"] = list()
    for thr_idx in range(len(thresholds)):
        stat_book["image_log"].append(list())
        stat_book["images_fp"].append(list())
        stat_book["images_fn_obstacle"].append(list())
        stat_book["images_fn_correct"].append(list())
        stat_book["conf_obstacle"].append({"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        stat_book["conf_correct"].append({"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    stat_book["auroc_data"] = np.zeros((1001, 2))

    overall_max = 1
    print(f"Overall Max: {overall_max}")


    with torch.no_grad():
        for idx, (image_fishy, target_fishy, image_orig, target_orig) in enumerate(metric_logger.log_every(data_loader, 100, header)):
            print(f"Image {idx} ...")
            if idx > args.max_num_images:
                break
            # Go over fishy:
            for mode in ["fishy", "orig"]:
                #time_prep_start = time.time()
                # Prepare everything
                if mode == "fishy":
                    image, target_seg = image_fishy.to(device), target_fishy.to(device)
                elif mode == "orig":
                    image, target_seg = image_orig.to(device), target_orig.to(device)
                target_seg_orig = target_orig.clone().to(device)

                # Mask for evaluation (discard background)
                if args.rails_only > 0:
                    evaluation_mask = target_seg_orig == 1
                    target_seg[torch.logical_not(evaluation_mask)] = 0
                    evaluation_mask = evaluation_mask.squeeze()
                else:
                    evaluation_mask = torch.logical_or(target_seg == 2, target_seg == 1).squeeze()

                # Visualize original image
                if args.g_act == "tanh":
                    image_target_ae, _ = presets.denormalize_tanh(image, image)  # (-1, 1)
                    image_vis, _ = presets.re_convert_tanh(image_target_ae, image_target_ae)
                else:
                    image_target_ae, _ = presets.denormalize(image, image)  # (0, 1)
                    image_vis, _ = presets.re_convert(image_target_ae, image_target_ae)
                VIS_INPUT = image_vis

                if ae_model and args.ae_model != "patchsegmodellight":
                    # Run  AE inference
                    with torch.no_grad():
                        outputs = ae_model(image)
                    output_ae = outputs["out_aa"]

                    # Post-process AE image for PatchSeg
                    if args.g_act == "tanh":
                        image_ae = (output_ae / 2) + 0.5
                    else:
                        image_ae = output_ae
                    image_ae = torchvision.transforms.functional.normalize(image_ae, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

                    # Visualize AE image
                    if args.g_act == "tanh":
                        image_ae_vis, _ = presets.re_convert_tanh(output_ae, output_ae)  # no de-normalization
                    else:
                        image_ae_vis, _ = presets.re_convert(output_ae, output_ae)  # no de-normalization
                    VIS_OUTPUT = image_ae_vis
                else:
                    VIS_OUTPUT = None

                # Prepare input for PatchSeg model
                if "patch30" in args.model:
                    image_r1 = torch.zeros_like(image)
                    image_r1[::, :-27] = image[::, 27:]
                    image_r2 = torch.zeros_like(image)
                    image_r2[::, :-54] = image[::, 54:]
                    image_l1 = torch.zeros_like(image)
                    image_l1[::, 27:] = image[::, :-27]
                    image_l2 = torch.zeros_like(image)
                    image_l2[::, 54:] = image[::, :-54]
                    image_ae_r1 = torch.zeros_like(image_ae)
                    image_ae_r1[::, :-27] = image_ae[::, 27:]
                    image_ae_r2 = torch.zeros_like(image_ae)
                    image_ae_r2[::, :-54] = image_ae[::, 54:]
                    image_ae_l1 = torch.zeros_like(image_ae)
                    image_ae_l1[::, 27:] = image_ae[::, :-27]
                    image_ae_l2 = torch.zeros_like(image_ae)
                    image_ae_l2[::, 54:] = image_ae[::, :-54]
                    input_seg = torch.cat((image_l2, image_l1, image, image_r1, image_r2, image_ae_l2, image_ae_l1, image_ae, image_ae_r1, image_ae_r2), dim=1)
                elif "patch15" in args.model:
                    image_r1 = torch.zeros_like(image)
                    image_r1[::, :-27] = image[::, 27:]
                    image_r2 = torch.zeros_like(image)
                    image_r2[::, :-54] = image[::, 54:]
                    image_l1 = torch.zeros_like(image)
                    image_l1[::, 27:] = image[::, :-27]
                    image_l2 = torch.zeros_like(image)
                    image_l2[::, 54:] = image[::, :-54]
                    input_seg = torch.cat((image_l2, image_l1, image, image_r1, image_r2), dim=1)
                elif "patch6" in args.model:
                    input_seg = torch.cat((image, image_ae), dim=1)
                else:
                    input_seg = image

                # Inference
                if "patchclassmodel" in args.model:
                    with torch.no_grad():
                        output_seg = model(input_seg)["out"]
                        output_seg = nn.functional.softmax(output_seg, dim=1)
                        output_seg = output_seg[0, 0, ::]
                elif args.model == "deeplabv3_resnet50":
                    with torch.no_grad():
                        output_seg = model(input_seg)["out"]
                        output_seg = nn.functional.softmax(output_seg, dim=1)
                        output_seg = output_seg[0, 0, ::]
                elif args.model == "mse":
                    if args.g_act == "tanh":
                        image_target_ae = (image_target_ae / 2) + 0.5
                        output_ae = (output_ae / 2) + 0.5
                    output_seg = torch.squeeze(torch.sqrt(torch.square(image_target_ae - output_ae)))
                    output_seg = torch.mean(output_seg, dim=0)
                elif args.model == "ssim":
                    ssim = SSIM(11)
                    if args.g_act == "tanh":
                        image_target_ae = (image_target_ae / 2) + 0.5
                        output_ae = (output_ae / 2) + 0.5
                    output_seg = torch.squeeze(ssim(image_target_ae, output_ae))*2 # SSIM output is in range (0, 0.5)
                    output_seg = torch.mean(output_seg, dim=0)
                elif args.model == "patchsegmodellight" and args.ae_model == "patchsegmodellight": # Student Teacher
                    with torch.no_grad():
                        outputs_teacher = ae_model(input_seg)["descriptor"]
                        normalized_teacher = F.normalize(outputs_teacher, mean=mean, std=std).clone().detach()
                        outputs_student = model(input_seg)["descriptor"]
                        output_seg = torch.squeeze(torch.sqrt(torch.square(normalized_teacher - outputs_student)))
                        # print(f"Output_seg: {output_seg.shape}, max: {torch.max(output_seg)}, min: {torch.min(output_seg)}")
                        # Make sure output_seg is in range (0, 1)
                        output_seg = torch.mean(output_seg, dim=0)

                # Make sure segmentation outputs are in range (0, 1):
                output_seg = output_seg / overall_max
                # Check if segmentation outputs are in range (0, 1):
                if torch.max(output_seg) > 1 or torch.min(output_seg) < 0:
                    print("ERROR: Output segmentation out of range!")
                    return
                #time_prep_end = time.time()
                #print(f"Time Inference: {time_prep_end-time_prep_start}")

                # Compute whether an obstacle can be found in groundtruth:
                target_seg_masked = target_seg.clone().type(torch.FloatTensor)
                target_seg_masked[target_seg == 1] = 0
                target_seg_masked[target_seg == 2] = 1
                kernel = torch.tensor(
                    np.ones((args.k_d, args.k_d)) * 1 / (args.k_d * args.k_d)).view(1, 1,
                                                                                                                args.k_d,
                                                                                                                args.k_d).type(
                    torch.FloatTensor)
                patch_density_target = torch.nn.functional.conv2d(target_seg_masked, kernel, padding='same')
                max_patch_density_target = torch.max(patch_density_target)
                if max_patch_density_target > 0.3:
                    has_obstacle = 1
                    # Get bounding box
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
                                if target_seg[0, 0, i, j] == 2:
                                    stat_book["auroc_data"][val, 1] += 1
                                else:
                                    stat_book["auroc_data"][val, 0] += 1

                for thr_idx, thr in enumerate(thresholds):
                    visualization_images = list()
                    visualization_images.append(VIS_INPUT)
                    if VIS_OUTPUT:
                        visualization_images.append(VIS_OUTPUT)
                    #time_loc_start = time.time()
                    # Compute whether an obstacle can be found in seg output based on patch density:
                    output_seg_masked = output_seg.clone()
                    output_seg_masked[torch.logical_not(evaluation_mask)] = 0
                    patch_density = torch.nn.functional.conv2d(output_seg_masked.unsqueeze(0).unsqueeze(1), kernel, padding='same')
                    max_patch_density = torch.max(patch_density)
                    patch_density[patch_density <= thr] = 0
                    patch_density.squeeze()
                    if max_patch_density > thr:
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
                    #time_loc_end = time.time()
                    #print(f"Loc time: {time_loc_end - time_loc_start}")

                    # Check whether found obstacle was correct
                    if found_obstacle == 1 and has_obstacle == 1 and l_bb < centroid_x < r_bb and u_bb < centroid_y < d_bb:
                        found_correct = True
                    else:
                        found_correct = False

                    # Now fill statistics book
                    image_data = {"idx": idx, "mode": mode, "has_obstacle": has_obstacle, "found_obstacle": found_obstacle, "found_correct": found_correct}
                    stat_book["image_log"][thr_idx].append(image_data)

                    # Fill confusion matrices:
                    if has_obstacle == 1:
                        if found_obstacle == 1:
                            stat_book["conf_obstacle"][thr_idx]["tp"] += 1
                        else:
                            stat_book["conf_obstacle"][thr_idx]["fn"] += 1
                            stat_book["images_fn_obstacle"][thr_idx].append(idx)
                        if found_correct == 1:
                            stat_book["conf_correct"][thr_idx]["tp"] += 1
                        else:
                            stat_book["conf_correct"][thr_idx]["fn"] += 1
                            stat_book["images_fn_correct"][thr_idx].append(idx)
                    else:  # if there is no obstacle, it does not matter if it was classified correctly or not
                        if found_obstacle == 1:
                            stat_book["conf_obstacle"][thr_idx]["fp"] += 1
                            stat_book["conf_correct"][thr_idx]["fp"] += 1
                            stat_book["images_fp"][thr_idx].append(idx)
                        else:
                            stat_book["conf_obstacle"][thr_idx]["tn"] += 1
                            stat_book["conf_correct"][thr_idx]["tn"] += 1

                    # Visualize Output Patchseg
                    output_seg_vis_gray = output_seg / torch.max(output_seg) * 255
                    output_seg_vis_gray = presets.torch_mask_to_pil(output_seg_vis_gray)
                    output_seg_vis = Image.new("RGB", output_seg_vis_gray.size)
                    output_seg_vis.paste(output_seg_vis_gray)
                    visualization_images.append(output_seg_vis)

                    # Visualize Patch Density
                    patch_density_vis_gray = patch_density * 255
                    patch_density_vis_gray = presets.torch_mask_to_pil(patch_density_vis_gray)
                    patch_density_vis = Image.new("RGB", patch_density_vis_gray.size)
                    patch_density_vis.paste(patch_density_vis_gray)
                    draw = ImageDraw.Draw(patch_density_vis)
                    if found_obstacle == 1:
                        if found_correct == 1:
                            draw.ellipse((centroid_x-3 , centroid_y-3, centroid_x+3 , centroid_y+3), fill="green")
                        else:
                            draw.ellipse((centroid_x - 3, centroid_y - 3, centroid_x + 3, centroid_y + 3), fill="red")
                    if has_obstacle == 1:
                        draw.rectangle((l_bb, u_bb, r_bb, d_bb), outline="blue")
                    visualization_images.append(patch_density_vis)

                    # Visualized output seg masked
                    output_seg_masked_vis_gray = output_seg
                    output_seg_masked_vis_gray[torch.logical_not(evaluation_mask)] = 0.5
                    output_seg_masked_vis_gray = output_seg_masked_vis_gray * 255
                    output_seg_masked_vis_gray = presets.torch_mask_to_pil(output_seg_masked_vis_gray)
                    output_seg_masked_vis = Image.new("RGB", output_seg_masked_vis_gray.size)
                    output_seg_masked_vis.paste(output_seg_masked_vis_gray)
                    draw = ImageDraw.Draw(output_seg_masked_vis)
                    if found_obstacle == 1:
                        draw.text((0, 0), f"Max patch %: {max_patch_density:.2f} --> 1", (0, 255, 0))
                    else:
                        draw.text((0, 0), f"Max patch %: {max_patch_density:.2f} --> 0", (255, 0, 0))
                    visualization_images.append(output_seg_masked_vis)

                    # Visualize target segmentation
                    target_obs_seg_vis_gray = torch.zeros_like(target_seg)
                    target_obs_seg_vis_gray[target_seg == 2] = 255 # obstacle white
                    target_obs_seg_vis_gray[target_seg == 0] = 127 # background gray
                    target_obs_seg_vis_gray = presets.torch_mask_to_pil(target_obs_seg_vis_gray)
                    target_obs_seg_vis = Image.new("RGB", target_obs_seg_vis_gray.size)
                    target_obs_seg_vis.paste(target_obs_seg_vis_gray)
                    draw = ImageDraw.Draw(target_obs_seg_vis)
                    if has_obstacle == 1:
                        draw.text((0, 0), f"Max patch %: {max_patch_density_target:.2f} --> 1", (0, 255, 0))
                    else:
                        draw.text((0, 0), f"Max patch %: {max_patch_density_target:.2f} --> 0", (255, 0, 0))
                    visualization_images.append(target_obs_seg_vis)

                    # Stack horizontally
                    if args.theta_visualize == thr:
                        img_row = np.hstack((np.asarray(i) for i in visualization_images))
                        image_final = Image.fromarray(img_row)
                        image_final.save(os.path.join(vis_path, f"Thr{int(thr*100)}_{idx:04}_{mode}_visualization.jpeg"), format="jpeg")

        # Compute global metrics:
        fpr_list, tpr_list, thr_list, auc = roc_curve(stat_book["auroc_data"])
        roc_auc = auc(fpr_list, tpr_list)
        fig, ax = plt.subplots(1, 1)
        ax.plot(fpr_list, tpr_list, label='ROC curve (area = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC')
        ax.legend(loc="lower right")
        fig.savefig(os.path.join(vis_path, f"ROC.pdf"))

        output_string = ""

        f1_scores = list()

        # Print confusion matrices:
        for thr_idx, thr in enumerate(thresholds):
            for key in ["conf_obstacle", "conf_correct"]:
                tp = stat_book[key][thr_idx]["tp"]
                fp = stat_book[key][thr_idx]["fp"]
                tn = stat_book[key][thr_idx]["tn"]
                fn = stat_book[key][thr_idx]["fn"]
                tot = tp + fp + tn + fn
                pp = tp + fp
                pn = tn + fn
                p = tp + fn
                n = tn + fp
                if p != 0:
                    tpr = tp / (tp + fn) # True Positive Rate (Sensitivity, Recall): ratio of (correct) detection if there was an obstacle (SAFETY-CRITICAL)
                else:
                    tpr = 99999
                if pn != 0:
                    npv = tn / (fn + tn) # Negative Predictive Value: ratio of correct non-detections if there was no detection (SAFETY-CRITICAL)
                else:
                    npv = 99999
                if n != 0:
                    tnr = tn / (fp + tn) # True Negative Rate (Specificity, 1-FPR): ratio of (correct) non-detection if there was no obstacle
                else:
                    tnr = 99999
                if pp != 0:
                    ppv = tp / (tp + fp) # Positive Predictive Value (Precision): ratio of correct detection if there was a detection
                else:
                    ppv = 99999
                if tpr + ppv != 0:
                    f1 = 2 * (tpr * ppv) / (tpr + ppv) # F1 score
                else:
                    f1 = -1
                if key == "conf_correct":
                    f1_scores.append(f1)

                output_string += f"CONFMAT METRICS FOR {key} (Obstacle Patch Density Threshold: {thr}):\n\n"
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
                output_string += f"Image Ids for False Positive: {stat_book['images_fp'][thr_idx]}\n"
                output_string += f"Image Ids for False Negative Obstacle: {stat_book['images_fn_obstacle'][thr_idx]}\n"
                output_string += f"Image Ids for False Negative Correct: {stat_book['images_fn_correct'][thr_idx]}\n"
                output_string += f"\n\n"

        max_f1 = max(f1_scores)
        max_f1_idx = f1_scores.index(max_f1)
        max_f1_thr = thresholds[max_f1_idx]
        output_string += f"\n\nFINAL RESULTS:\n\n"
        output_string += f"AUROC: {roc_auc:.3f}\n"
        output_string += f"Max F1: {max_f1} with threshold {max_f1_thr}\n\n"
        output_string += f"\tfrom {f1_scores}\n"
        output_string += f"Image Ids for False Positive: {stat_book['images_fp'][max_f1_idx]}\n"
        output_string += f"Image Ids for False Negative Correct: {stat_book['images_fn_correct'][max_f1_idx]}\n"
        output_string += f"\n\n"

        print(output_string)
        with open(os.path.join(vis_path, "output.txt"), "w") as file:
            file.write(output_string)
        with open(os.path.join(vis_path, "args.txt"), 'w') as file:
            file.write(json.dumps(vars(args)))
    return

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

    print(args)
    device = torch.device(args.device)
    print(f"Running on device: {device}")

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    if "Railsem19" in args.data_path:
        dataset_test = RailSem19CroppedDatasetLikeFishyrails(args.data_path, mode="train")
    else:
        dataset_test = FishyrailsCroppedDataset(args.data_path)
    print("Dataset loaded.")

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=1
    )

    print("Dataloader created.")

    # Gan model
    if args.ae_model == "AeSegParam02_8810":
        ae_model = AeSegParam02(c_seg=8, c_ae=8, c_param=1, mode="none", ratio=args.color_space_ratio, act=args.g_act)
    elif args.ae_model == "AeSegParam02_8410":
        ae_model = AeSegParam02(c_seg=8, c_ae=4, c_param=1, mode="none", ratio=args.color_space_ratio, act=args.g_act)
    elif args.ae_model == "AeSegParam02_8210":
        ae_model = AeSegParam02(c_seg=8, c_ae=2, c_param=1, mode="none", ratio=args.color_space_ratio, act=args.g_act)
    elif args.ae_model == "AeSegParam02_8110":
        ae_model = AeSegParam02(c_seg=8, c_ae=1, c_param=1, mode="none", ratio=args.color_space_ratio, act=args.g_act)
    elif args.ae_model == "patchsegmodellight":
        ae_model = PatchSegModelLight(in_channels=3, out_channels=512, stages=args.stages, patch_only=False)
    else:
        ae_model = None
        print(f"No autoencoder used!")

    if ae_model:
        ae_model.to(device)
        ae_checkpoint = torch.load(args.ae_checkpoint, map_location="cpu")
        ae_model.load_state_dict(ae_checkpoint["model"], strict=False)
        print("AE Model loaded.")
        if args.ae_model == "patchsegmodellight" and args.model == "patchsegmodellight":
            args.mean_std_dir = os.path.dirname(args.ae_checkpoint)
            args.mean_std_suffix = os.path.basename(args.ae_checkpoint[:-4])
            with open(os.path.join(args.mean_std_dir, f"{args.mean_std_suffix}_mean.npy"), "rb") as file:
                mean = np.load(file)
            with open(os.path.join(args.mean_std_dir, f"{args.mean_std_suffix}_std.npy"), "rb") as file:
                std = np.load(file)
            mean = torch.from_numpy(mean).to(device)
            print(f"Mean shape: {mean.shape}")
            std = torch.from_numpy(std).to(device)
            print(f"Std shape: {std.shape}")
        else:
            mean = None
            std = None
    else:
        mean = None
        std = None

    # Segmentation model
    if args.model == "deeplabv3_resnet50":
        model = torchvision.models.segmentation.__dict__[args.model](
                pretrained=False,
                pretrained_backbone=False,
                num_classes=2,
                aux_loss=False,
            )
        #model.backbone.conv1 = nn.Conv2d(6, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
    elif args.model == "patchsegmodellight_patch30":
        model = PatchSegModelLight(stages=args.stages, in_channels=30)
    elif args.model == "patchclassmodel_patch30":
        model = PatchClassModel(stages=args.stages, in_channels=30)
    elif args.model == "patchclassmodel_patch15":
        model = PatchClassModel(stages=args.stages, in_channels=15)
    elif args.model == "patchclassmodel_patch6":
        model = PatchClassModel(stages=args.stages, in_channels=6)
    elif args.model == "patchclassmodel_patch3":
        model = PatchClassModel(stages=args.stages, in_channels=3)
    elif args.model == "patchsegmodellight":
        model = PatchSegModelLight(in_channels=3, out_channels=512, stages=args.stages, patch_only=False)
    else:
        model = None
        print("No seg model!")

    if model:
        model.to(device)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)

    evaluate(args, ae_model, model, data_loader_test, device=device, num_classes=num_classes, vis_path=args.output_path, mean=mean, std=std)
    return


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--data_path", default="./datasets/FishyrailsCroppedDebug/FishyrailsCroppedDebug.h5", type=str, help="dataset path")
    parser.add_argument("--model", default="deeplabv3_resnet50", type=str, help="model name")
    parser.add_argument("--output_path", default="./evaluation", type=str, help="output directory")
    parser.add_argument("--device", default="cpu", type=str, help="device (Use cuda or cpu)")
    parser.add_argument("--checkpoint", default="./trained_models/patchclass_model_10.pth", type=str,
                        help="path of checkpoint")
    parser.add_argument("--ae_checkpoint", default="./trained_models/ganaesegparam02_8810_01000_017_model_199.pth", type=str, help="path of checkpoint")
    parser.add_argument("--ae_model", default="AeSegParam02_8810", type=str, help="Autoencoder Type")
    parser.add_argument("--color_space_ratio", default=0.1, type=float, help="color space ratio for each channel, NOT relevant for our experiments")
    parser.add_argument("--max_num_images", default=4000, type=int, help="max number of images to be evaluated")
    parser.add_argument("--g_act", default="tanh", type=str, help="generator activation")
    parser.add_argument("--theta_visualize", default=0.0, type=float, help="which obstacle threshold to visualize")
    parser.add_argument(
        "--seg_pretrained",
        dest="seg_pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument("--stages", default=1, type=int, help="Number of stages of the Patch Classification network. Stage 0 corresponds to patch size 13, 1 to 21, 2 to 29, 3 to 35, and 4 to 51.")
    parser.add_argument("--k_d", default=21, type=int, help="patch density size k_d")
    parser.add_argument("--rails_only", default=1, type=int, help="whether to evaluate on rails only")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
