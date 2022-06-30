import datetime
import os
import time
import numpy as np

import presets
import torch
import torch.utils.data
import utils
import json
from PIL import Image
from PIL import ImageDraw

from dataset import FishyrailsCroppedDataset, RailSem19CroppedDatasetLikeFishyrails, RailSem19CroppedDatasetSegmentation
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from patchclass_networks import PatchSegModelLight
from torchvision.transforms import functional as F

def get_transform(train, args):
    if train:
        # base_size is normal size of image, is then up/downsampled by 0.5 / 2, then image is cropped to crop_size
        return presets.SegmentationPresetTrain(base_size=224, crop_size=224)
    else:
        # image is rescaled to base_size
        return presets.SegmentationPresetEval(base_size=224)

def compute_mean(args, teacher, student1, student2, student3, data_loader, device, mean_teacher, std_teacher):
    teacher.eval()
    student1.eval()
    student2.eval()
    student3.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Mean:"
    mean_e_acc = 0
    mean_v_acc = 0
    with torch.no_grad():
        for idx, (image, target) in enumerate(metric_logger.log_every(data_loader, 100, header)):
            print(f"Mean: {idx}/{len(data_loader)}")
            image = image.to(device)
            outputs_teacher = teacher(image)["descriptor"]
            normalized_teacher = F.normalize(outputs_teacher, mean=mean_teacher, std=std_teacher).clone().detach()
            outputs_student1 = student1(image)["descriptor"]
            outputs_student2 = student2(image)["descriptor"]
            outputs_student3 = student3(image)["descriptor"]
            output_e_students = 1 / 3 * (outputs_student1 + outputs_student2 + outputs_student3)
            output_e = torch.squeeze(torch.square(normalized_teacher - output_e_students))
            output_e = torch.sum(output_e, dim=0)
            output_v_mean = torch.sum(torch.squeeze(torch.square(output_e_students)), dim=0)
            output_v_student1 = torch.sum(torch.squeeze(torch.square(outputs_student1)), dim=0)
            output_v_student2 = torch.sum(torch.squeeze(torch.square(outputs_student2)), dim=0)
            output_v_student3 = torch.sum(torch.squeeze(torch.square(outputs_student3)), dim=0)
            output_v = 1 / 3 * (output_v_student1 + output_v_student2 + output_v_student3) - output_v_mean

            mean_e = torch.mean(output_e)
            mean_v = torch.mean(output_v)
            #diff_raw = torch.squeeze(torch.square(outputs_student - normalized_teacher))
            #diff_raw = torch.mean(diff_raw, dim=0)
            mean_e_acc += mean_e.cpu()
            mean_v_acc += mean_v.cpu()
        print(f"Dataloader length: {len(data_loader)}")
        mean_e = mean_e_acc / len(data_loader)
        mean_v = mean_v_acc / len(data_loader)
        print(f"Mean e: {mean_e}")
        print(f"Mean v: {mean_v}")
    return mean_e, mean_v

def compute_std(args, teacher, student1, student2, student3, data_loader, device, mean_e, mean_v, mean_teacher, std_teacher):
    teacher.eval()
    student1.eval()
    student2.eval()
    student3.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Std:"
    std_e_acc = 0
    std_v_acc = 0
    mean_e = mean_e.view((1, 1))
    mean_e_repeated = mean_e.repeat((224, 224)).to(device)
    mean_v = mean_v.view((1, 1))
    mean_v_repeated = mean_v.repeat((224, 224)).to(device)
    print(f"Repeated mean e shape: {mean_e_repeated.shape}")
    print(f"Repeated mean v shape: {mean_v_repeated.shape}")
    with torch.no_grad():
        for idx, (image, target) in enumerate(metric_logger.log_every(data_loader, 100, header)):
            print(f"Std: {idx}/{len(data_loader)}")
            image = image.to(device)
            outputs_teacher = teacher(image)["descriptor"]
            normalized_teacher = F.normalize(outputs_teacher, mean=mean_teacher, std=std_teacher).clone().detach()
            outputs_student1 = student1(image)["descriptor"]
            outputs_student2 = student2(image)["descriptor"]
            outputs_student3 = student3(image)["descriptor"]
            output_e_students = 1 / 3 * (outputs_student1 + outputs_student2 + outputs_student3)
            output_e = torch.squeeze(torch.square(normalized_teacher - output_e_students))
            output_e = torch.sum(output_e, dim=0)
            output_v_mean = torch.sum(torch.squeeze(torch.square(output_e_students)), dim=0)
            output_v_student1 = torch.sum(torch.squeeze(torch.square(outputs_student1)), dim=0)
            output_v_student2 = torch.sum(torch.squeeze(torch.square(outputs_student2)), dim=0)
            output_v_student3 = torch.sum(torch.squeeze(torch.square(outputs_student3)), dim=0)
            output_v = 1 / 3 * (output_v_student1 + output_v_student2 + output_v_student3) - output_v_mean
            std_e = torch.sum(torch.square(output_e - mean_e_repeated))
            std_v = torch.sum(torch.square(output_v - mean_v_repeated))
            #diff_raw = torch.squeeze(torch.square(outputs_student - normalized_teacher))
            #diff_raw = torch.mean(diff_raw, dim=0)
            std_e_acc += std_e.cpu()
            std_v_acc += std_v.cpu()
        print(f"Dataloader length: {len(data_loader)}")
        std_e = std_e_acc / (len(data_loader) * 224*224)
        std_e = torch.sqrt(std_e)
        std_v = std_v_acc / (len(data_loader) * 224*224)
        std_v = torch.sqrt(std_v)
        print(f"Std e Total: {std_e}")
        print(f"Std v Total: {std_v}")
    return std_e, std_v



def evaluate(args, teacher, student1, student2, student3, data_loader, device, vis_path=None, mean=None, std=None, overall_max=1):
    teacher.eval()
    student1.eval()
    student2.eval()
    student3.eval()

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

    print(f"Overall Max: {overall_max}")


    with torch.no_grad():
        for idx, (image_fishy, target_fishy, image_orig, target_orig) in enumerate(metric_logger.log_every(data_loader, 100, header)):
            print(f"Image {idx} ...")
            if idx > args.max_num_images:
                break
            # Go over fishy:
            for mode in ["fishy", "orig"]:
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
                image_target_ae, _ = presets.denormalize(image, image)  # (0, 1)
                image_vis, _ = presets.re_convert(image_target_ae, image_target_ae)
                VIS_INPUT = image_vis

                # Dummy assignment
                VIS_OUTPUT = None
                input_seg = image

                # Inference
                with torch.no_grad():
                    outputs_teacher = teacher(input_seg)["descriptor"]
                    normalized_teacher = F.normalize(outputs_teacher, mean=mean["teacher"], std=std["teacher"]).clone().detach()
                    outputs_student1 = student1(input_seg)["descriptor"]
                    outputs_student2 = student2(input_seg)["descriptor"]
                    outputs_student3 = student3(input_seg)["descriptor"]
                    output_e_students = 1/3 * (outputs_student1 + outputs_student2 + outputs_student3)
                    output_e = torch.squeeze(torch.square(normalized_teacher - output_e_students))
                    output_e = torch.sum(output_e, dim=0)
                    output_e_normalized = torch.abs((output_e - mean["e"]) / std["e"])
                    output_v_mean = torch.sum(torch.squeeze(torch.square(output_e_students)), dim=0)
                    output_v_student1 = torch.sum(torch.squeeze(torch.square(outputs_student1)), dim=0)
                    output_v_student2 = torch.sum(torch.squeeze(torch.square(outputs_student2)), dim=0)
                    output_v_student3 = torch.sum(torch.squeeze(torch.square(outputs_student3)), dim=0)
                    output_v = 1/3 * (output_v_student1 + output_v_student2 + output_v_student3) - output_v_mean
                    output_v_normalized = torch.abs((output_v - mean["v"]) / std["v"])
                    output_seg = output_e_normalized + output_v_normalized


                # Make sure segmentation outputs are in range (0, 1):
                output_seg[output_seg > overall_max] = overall_max
                output_seg = output_seg / overall_max
                # Check if segmentation outputs are in range (0, 1):
                if torch.max(output_seg) > 1 or torch.min(output_seg) < 0:
                    print("ERROR: Output segmentation out of range!")
                    return

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
                    # Compute whether an obstacle can be found in seg output based on patch density:
                    output_seg_masked = output_seg.clone()
                    output_seg_masked[torch.logical_not(evaluation_mask)] = 0
                    patch_density = torch.nn.functional.conv2d(output_seg_masked.unsqueeze(0).unsqueeze(1).cpu(), kernel, padding='same')
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

    # use our dataset and defined transformations
    if "Railsem19" in args.data_path_test:
        dataset_test = RailSem19CroppedDatasetLikeFishyrails(args.data_path_test, mode="train")
    else:
        dataset_test = FishyrailsCroppedDataset(args.data_path_test)
    print("Dataset loaded.")

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=1
    )
    print(f"Dataloader test: {len(data_loader_test)}")

    # Validation Dataset
    dataset_val = RailSem19CroppedDatasetSegmentation(args.data_path_val, get_transform(train=False, args=args),
                                                       mode="val", train_fraction=args.train_fraction)
    val_sampler = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, sampler=val_sampler, num_workers=1, collate_fn=utils.collate_fn
    )
    print(f"Dataloader val: {len(data_loader_val)}")
    print("Dataloader created.")

    # Models
    teacher = PatchSegModelLight(in_channels=3, out_channels=128, stages=args.stages, patch_only=False).to(device)
    student1 = PatchSegModelLight(in_channels=3, out_channels=128, stages=args.stages, patch_only=False).to(device)
    student2 = PatchSegModelLight(in_channels=3, out_channels=128, stages=args.stages, patch_only=False).to(device)
    student3 = PatchSegModelLight(in_channels=3, out_channels=128, stages=args.stages, patch_only=False).to(device)

    teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu")
    teacher.load_state_dict(teacher_checkpoint["model"], strict=False)
    student1_checkpoint = torch.load(args.student1_checkpoint, map_location="cpu")
    student1.load_state_dict(student1_checkpoint["model"], strict=False)
    student2_checkpoint = torch.load(args.student2_checkpoint, map_location="cpu")
    student2.load_state_dict(student2_checkpoint["model"], strict=False)
    student3_checkpoint = torch.load(args.student3_checkpoint, map_location="cpu")
    student3.load_state_dict(student3_checkpoint["model"], strict=False)

    # Load Mean and Std (pre-computed)
    args.mean_std_dir = os.path.dirname(args.teacher_checkpoint)
    args.mean_std_suffix = os.path.basename(args.teacher_checkpoint[:-4])
    with open(os.path.join(args.mean_std_dir, f"{args.mean_std_suffix}_mean.npy"), "rb") as file:
        mean = np.load(file)
    with open(os.path.join(args.mean_std_dir, f"{args.mean_std_suffix}_std.npy"), "rb") as file:
        std = np.load(file)
    mean_teacher = torch.from_numpy(mean).to(device)
    print(f"Mean shape: {mean.shape}")
    std_teacher = torch.from_numpy(std).to(device)
    print(f"Std shape: {std.shape}")

    # Do computations
    mean_e, mean_v = compute_mean(args, teacher, student1, student2, student3, data_loader_val, device, mean_teacher, std_teacher)
    std_e, std_v = compute_std(args, teacher, student1, student2, student3, data_loader_val, device, mean_e, mean_v, mean_teacher, std_teacher)
    mean = {"teacher": mean_teacher, "e": mean_e, "v": mean_v}
    std = {"teacher": std_teacher, "e": std_e, "v": std_v}
    evaluate(args, teacher, student1, student2, student3, data_loader_test, device=device, vis_path=args.output_path, mean=mean, std=std, overall_max=args.overall_max)
    return


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)
    parser.add_argument("--data_path_val", default="./datasets/Railsem19CroppedDebug/Railsem19CroppedDebug.h5", type=str, help="val dataset path")
    parser.add_argument("--data_path_test", default="./datasets/FishyrailsCroppedDebug/FishyrailsCroppedDebug.h5", type=str, help="test dataset path")
    parser.add_argument("--output_path", default="./evaluation", type=str, help="output directory")
    parser.add_argument("--device", default="cpu", type=str, help="device (Use cuda or cpu)")
    parser.add_argument("--teacher_checkpoint", default="./trained_models/teacher.pth", type=str,
                        help="path of teacher checkpoint")
    parser.add_argument("--student1_checkpoint", default="./trained_models/student1.pth", type=str,
                        help="path of student1 checkpoint")
    parser.add_argument("--student2_checkpoint", default="./trained_models/student2.pth", type=str,
                        help="path of student2 checkpoint")
    parser.add_argument("--student3_checkpoint", default="./trained_models/student3.pth", type=str,
                        help="path of student3 checkpoint")
    parser.add_argument("--max_num_images", default=4000, type=int, help="max number of images to be evaluated")
    parser.add_argument("--theta_visualize", default=0.0, type=float, help="which threshold theta to visualize")
    parser.add_argument("--stages", default=7, type=int, help="number of stages of neural network, stages=1 corresponds to patch_size of 17, stages=2 to 33, stage=3 to 65")
    parser.add_argument("--k_d", default=29, type=int, help="obstacle threshold (patch density)")
    parser.add_argument("--rails_only", default=1, type=int, help="whether to evaluate on rails only")
    parser.add_argument("--train_fraction", default=0.9, type=float, help="fraction of train images")
    parser.add_argument("--overall_max", default=5.0, type=float, help="maximum value to consider in segmentation map")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
