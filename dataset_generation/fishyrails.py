import scipy as sp
import cv2
import os
import random
import zipfile
from skimage.filters import gaussian as gaussian_filter
from skimage.util import img_as_ubyte, img_as_float
import math
import numpy as np
import argparse
import glob
from PIL import Image
import json
from utils import compute_midpoints
import xml2dict



def main(args):
    # Prepare output directories
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    if not os.path.exists(OUTPUT_MASK_PATH):
        os.makedirs(OUTPUT_MASK_PATH)
    if not os.path.exists(OUTPUT_ORIG_PATH):
        os.makedirs(OUTPUT_ORIG_PATH)
    if not os.path.exists(OUTPUT_ORIG_MASK_PATH):
        os.makedirs(OUTPUT_ORIG_MASK_PATH)
    if not os.path.exists(OUTPUT_OBSTACLE_PATH):
        os.makedirs(OUTPUT_OBSTACLE_PATH)
    if not os.path.exists(OUTPUT_JSON_PATH):
        os.makedirs(OUTPUT_JSON_PATH)

    # Get list of images and segmentations and check if everything is alright
    # RailSem19
    rail_image_regex = os.path.join(RAIL_IMAGE_PATH, f"*{RAIL_IMAGE_EXTENSION}")
    rail_image_list = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(rail_image_regex)]
    rail_image_list = [x for x in rail_image_list if args.start_image_name <= x <= args.end_image_name]
    rail_image_list.sort()
    rail_seg_regex = os.path.join(RAIL_SEG_PATH, f"*{RAIL_SEG_EXTENSION}")
    rail_seg_list = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(rail_seg_regex)]
    rail_seg_list = [x for x in rail_seg_list if args.start_image_name <= x <= args.end_image_name]
    rail_seg_list.sort()
    print(f"Found {len(rail_image_list)} rail images from {RAIL_IMAGE_PATH})")
    print(f"Found {len(rail_seg_list)} rail segmentations from {RAIL_SEG_PATH})")
    if len(rail_image_list) != len(rail_seg_list):
        print("Number of rail images and segmentations does not match!")
        return -1
    # Obstacles
    obstacle_image_regex = os.path.join(OBSTACLE_IMAGE_PATH, f"*{OBSTACLE_IMAGE_EXTENSION}")
    obstacle_image_list = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(obstacle_image_regex)]
    obstacle_image_list.sort()
    obstacle_class_seg_regex = os.path.join(OBSTACLE_CLASS_SEG_PATH, f"*{OBSTACLE_SEG_EXTENSION}")
    obstacle_class_seg_list = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(obstacle_class_seg_regex)]
    obstacle_class_seg_list.sort()
    obstacle_obj_seg_regex = os.path.join(OBSTACLE_OBJ_SEG_PATH, f"*{OBSTACLE_SEG_EXTENSION}")
    obstacle_obj_seg_list = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(obstacle_obj_seg_regex)]
    obstacle_obj_seg_list.sort()
    print(f"Found {len(obstacle_image_list)} obstacle images from {OBSTACLE_IMAGE_PATH})")
    print(f"Found {len(obstacle_class_seg_list)} class obstacle segmentations from {OBSTACLE_CLASS_SEG_PATH})")
    print(f"Found {len(obstacle_obj_seg_list)} object obstacle segmentations from {OBSTACLE_OBJ_SEG_PATH})")
    if len(obstacle_class_seg_list) != len(obstacle_obj_seg_list):
        print("Number of obstacle images and segmentations does not match!")
        return -1
    # Prepare obstacles:
    obstacle_dict = prepare_obstacles(args, obstacle_class_seg_list)

    # Add obstacles to images
    counter = 0
    for image_name in rail_image_list:
        print(f"Enhancing image {image_name}")
        image_path = os.path.join(RAIL_IMAGE_PATH, f"{image_name}{RAIL_IMAGE_EXTENSION}")
        rail_seg_path = os.path.join(RAIL_SEG_PATH, f"{image_name}{RAIL_SEG_EXTENSION}")
        json_path = os.path.join(JSON_PATH, f"{image_name}{JSON_EXTENSION}")
        rail_image = np.array(Image.open(image_path))

        rail_image = cv2.cvtColor(rail_image, cv2.COLOR_RGB2BGR)

        rail_seg = np.array(Image.open(rail_seg_path))

        # remove images without actual rails
        if np.count_nonzero(np.isin(rail_seg, [12])) == 0:
            continue
        rail_seg = np.isin(rail_seg, [12, 17, 18, 3])  # rail-track, rail non-drivable, rail drivable
        rail_seg = rail_seg.astype(np.uint8)
        with open(json_path) as json_file:
            json_data = json.load(json_file)

        # Re-output json_file
        output_json_path = os.path.join(OUTPUT_JSON_PATH, f"{image_name}{JSON_EXTENSION}")
        with open(output_json_path, "w") as json_file:
            json.dump(json_data, json_file)


        # Output input mask
        output_image_pil = Image.fromarray(rail_seg)
        output_path = os.path.join(OUTPUT_ORIG_MASK_PATH, f"{image_name}.png")
        output_image_pil.save(output_path, mode='L')

        obstacle_positions = get_obstacle_location(args, rail_image, json_data)
        if not obstacle_positions:
            continue

        base_blob = dict()
        base_blob['rgb'] = rail_image
        base_blob['labels'] = rail_seg

        # Add obstacle to rails
        output_blob = pasting_v2(SEED, base_blob, obstacle_positions, obstacle_dict)

        output_rgb = output_blob['rgb']
        output_orig = output_blob['orig']
        output_labels = output_blob['labels']
        output_mask = output_blob['mask']

        # Output Fishy Image
        output_fishy = cv2.cvtColor(output_rgb, cv2.COLOR_BGR2RGB)
        output_image_pil = Image.fromarray(output_fishy)
        output_path = os.path.join(OUTPUT_PATH, f"{image_name}.png")
        output_image_pil.save(output_path, mode='RGB')

        # Output Original Image
        output_orig = cv2.cvtColor(output_orig, cv2.COLOR_BGR2RGB)
        output_image_pil = Image.fromarray(output_orig)
        output_path = os.path.join(OUTPUT_ORIG_PATH, f"{image_name}.png")
        output_image_pil.save(output_path, mode='RGB')

        # Output mask
        output_seg_pil = Image.fromarray(output_mask.astype(np.uint8))
        output_path = os.path.join(OUTPUT_MASK_PATH, f"{image_name}.png")
        output_seg_pil.save(output_path, mode='L')

        counter += 1
        if counter > args.max_images:
            break
    return 0

def get_obstacle_location(args, image, json_data):
    obstacle_positions = list()
    for obj in json_data["objects"]:
        if "polyline-pair" in obj:
            coords = obj["polyline-pair"]
            height, width, _ = image.shape
            midpoints, distances, _, _, _ = compute_midpoints(coords)
            # If desired: Discard rails too far from the lower image center (only keep main rails)
            if args.main_rail_only:
                is_main_rail = False
            else:
                # all rails are main rails
                is_main_rail = True
            for midpoint in midpoints:
                if abs(midpoint[0] - width / 2) < width / 6 and height - midpoint[1] < height / 10:
                    is_main_rail = True
            # only add midpoints of main rail where distance is large enough to potential positions and that midpoint
            # is far enough from boundary (at least distance * 1/2 crop_distance_factor
            if is_main_rail:
                for midpoint, distance in zip(midpoints, distances):
                    if distance >= args.min_distance:
                        if abs(width/2 - midpoint[0]) < width/2 - distance*args.crop_distance_factor / 2 and \
                                abs(height/2 - midpoint[1]) < height/2 - distance*args.crop_aspect_ratio * args.crop_distance_factor/2:
                            obstacle_positions.append((midpoint, distance))
    return obstacle_positions


def prepare_obstacles(args, obstacle_seg_list):
    # Cut obstacles from PascalVOC dataset
    counter = 0
    # Return obstacle images sorted by classes
    obstacle_dict = dict()
    for image_name in obstacle_seg_list:
        print(f"Processing obstacle image {image_name}")
        # Load obstacle image data (RGB, class and object segmentation)
        image_path = os.path.join(OBSTACLE_IMAGE_PATH, f"{image_name}{OBSTACLE_IMAGE_EXTENSION}")
        obj_seg_path = os.path.join(OBSTACLE_OBJ_SEG_PATH, f"{image_name}{OBSTACLE_SEG_EXTENSION}")
        class_seg_path = os.path.join(OBSTACLE_CLASS_SEG_PATH, f"{image_name}{OBSTACLE_SEG_EXTENSION}")
        xml_path = os.path.join(OBSTACLE_XML_PATH, f"{image_name}{OBSTACLE_XML_EXTENSION}")
        with open(xml_path) as xml_file:
            xml_dict = xml2dict.parse(xml_file)
            xml_objects = list()
            # wrap into list:
            xml_objs = xml_dict["annotation"]["object"]
            if not isinstance(xml_objs, list):
                xml_objs = [xml_objs]
            for obj in xml_objs:
                obj_dict = {
                    "name": obj["name"],
                    "pose": obj["pose"],
                    "truncated": obj["truncated"],
                    "difficult": obj["difficult"],
                    "bbox": np.array([int(obj["bndbox"]["xmin"]), int(obj["bndbox"]["ymin"]), int(obj["bndbox"]["xmax"]), int(obj["bndbox"]["ymax"])])
                }
                xml_objects.append(obj_dict)
        image = np.array(Image.open(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        obj_seg = np.array(Image.open(obj_seg_path)) # load images that way to get rid of color palette
        class_seg = np.array(Image.open(class_seg_path))
        classes = np.unique(class_seg)
        # Go over all relevant classes present in the image
        for c in classes:
            if c in OBSTACLE_CLASSES:
                obj_seg_c = obj_seg.copy()
                obj_seg_c[class_seg != c] = 0
                objects = list(np.unique(obj_seg_c))
                objects.remove(0)
                # extract objects of that class separately
                for i_o, o in enumerate(objects):
                    obj_seg_o = obj_seg_c == o
                    # Check if obstacle is truncated or difficult
                    if is_truncated(obj_seg_o, xml_objects):
                        continue
                    obj_seg_o = np.expand_dims(obj_seg_o.astype(np.uint8)*255, 2)
                    obstacle_output_image = np.concatenate((image, obj_seg_o), axis=2)

                    # Output a 4-channeled image image with object segmentation as alpha channel
                    obstacle_output_image = cv2.cvtColor(obstacle_output_image, cv2.COLOR_BGRA2RGBA)
                    obstacle_output_image_pil = Image.fromarray(obstacle_output_image)

                    obstacle_output_path = os.path.join(OUTPUT_OBSTACLE_PATH, f"obstacle_{c:02}_{i_o}_{image_name}.png")
                    obstacle_output_image_pil.save(obstacle_output_path, mode="RGBA")
                    # Fill output dict
                    if c not in obstacle_dict.keys():
                        obstacle_dict[c] = list()
                    obstacle_dict[c].append(obstacle_output_path)
                    counter += 1
                    #cv2.imwrite(os.path.join(OUTPUT_OBSTACLE_PATH, f"obstacle_{c:02}_{i_o}_{image_name}.png"), image)
        if counter > args.max_obstacles:
            break
    return obstacle_dict

def is_truncated(obj_seg, xml_objects):
    # Get bbox from segmentation
    y_list, x_list = np.where(obj_seg == 1)
    x_min = np.min(x_list)
    x_max = np.max(x_list)
    y_min = np.min(y_list)
    y_max = np.max(y_list)
    bbox = np.array([x_min, y_min, x_max, y_max])
    # Match bbox to closest bbox from xml annotations
    matched_idx = None
    matched_dist = 9999999999
    for idx, obj in enumerate(xml_objects):
        dist = (np.square(bbox - obj["bbox"])).mean()
        if dist < matched_dist:
            matched_idx = idx
            matched_dist = dist
    # check if truncated or difficult
    truncated = bool(int(xml_objects[matched_idx]["truncated"]))
    difficult = bool(int(xml_objects[matched_idx]["difficult"]))
    too_small = x_max - x_min < 50 and y_max - y_min < 50
    return truncated or difficult or too_small




def motion_blurred_image(img, steps):
    h, w, _ = img.shape
    images = []
    for i in range(steps):
        bigger_h = h + 2 * i
        bigger_w = math.floor(w * bigger_h / h)
        bigger_w += (bigger_w - w) % 2
        padding_w = (bigger_w - w) // 2
        bigger_img = cv2.resize(img, dsize=(bigger_w, bigger_h))
        images.append(bigger_img[i:h + i, padding_w:w + padding_w])
    return np.mean(np.stack(images, axis=0), axis=0)


def glow_effect(img, weight, steps, sigma):
    blurred = gaussian_filter(img, sigma=sigma, multichannel=True, preserve_range=True)
    for _ in range(steps):
        blurred = gaussian_filter(blurred, sigma=sigma, multichannel=True, preserve_range=True)
    # blurred = sp.ndimage.gaussian_filter(blurred, sigma=[5, 5, 1])
    # return blurred
    output_image = np.minimum(weight * blurred.astype(int) + img.astype(int), np.ones_like(img) * 255)
    # take care of overflow
    output_image[output_image > 255] = 255
    output_image[output_image < 0] = 0
    return output_image


def histogram_adaptation(img, reference_img):
    """following
    https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    Switched to YCCrCb space due to the following post:
    # https://stackoverflow.com/questions/15007304/histogram-equalization-not-working-on-color-image-opencv"""
    img = img.astype(np.uint8)
    reference_img = reference_img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    reference_img = cv2.cvtColor(reference_img[..., :3], cv2.COLOR_BGR2YCrCb)
    # for channel, col in enumerate(['r', 'g', 'b']):
    for channel, col in enumerate(['Y']):
        # get reference histogram of background image
        hist, _ = np.histogram(reference_img[..., channel].flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        inverse_mapping = np.empty(256)
        last_idx = 0
        for i in range(256):
            inverse_mapping[last_idx:cdf[i] + 1] = i
            last_idx = cdf[i]
        img[..., channel] = inverse_mapping[img[..., channel]]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    return img

def get_obj(obj_dict, obj_cat):
    # Get object by category
    obj_image_path = random.sample(obj_dict[obj_cat], 1)[0]
    obj = np.array(Image.open(obj_image_path))
    obj = cv2.cvtColor(obj, cv2.COLOR_RGBA2BGRA)
    # Get background
    obj_background = obj.copy()
    obj_background[:, :, 3] = 255  # fill alpha channel
    # Get alpha
    alpha = obj[..., 3:]

    # remove any clutter around the object of interest (with margin 5)
    h, w, _ = obj.shape
    obj_binary_mask = (alpha > 0).astype('int8')
    left_crop = next((i for i in range(w) if np.sum(obj_binary_mask[:, i + 5]) > 0))
    right_crop = next((i for i in range(w - 1, 0, -1)
                       if np.sum(obj_binary_mask[:, i - 5]) > 0))
    top_crop = next((i for i in range(h) if np.sum(obj_binary_mask[i + 5]) > 0))
    bottom_crop = next((i for i in range(h - 1, 0, -1)
                        if np.sum(obj_binary_mask[i - 5]) > 0))
    obj = obj[top_crop:bottom_crop + 1, left_crop:right_crop + 1]
    h, w, _ = obj.shape
    return obj, obj_background, h, w


def pasting_v2(seed, base_blob, positions, obj_dict):
    rand = np.random.RandomState(seed=seed)

    # Extract basic information from base image
    img_h, img_w, _ = base_blob['rgb'].shape
    labels = base_blob['labels'].copy()

    # Sample obstacle location from midpoints
    sampled_idx = random.sample(range(0, len(positions)), 1)[0]
    midpoint, d_rail = positions[sampled_idx]

    # Sample obstacle category + get respective size
    upper_cat = random.sample(SAMPLING_LIST, 1)[0]
    obj_cat = 99
    while obj_cat not in obj_dict.keys():
        obj_cat = random.sample(SAMPLING_DICT[upper_cat], 1)[0]
    cat_width = OBSTACLE_WIDTH_METER[obj_cat]

    # Get target obstacle width
    factor = 1.3
    ref_width = int(d_rail / 1.5 * cat_width)
    size_limits = [int(ref_width / factor), int(ref_width * factor)]
    target_w = int(rand.randint(*size_limits))

    # Repeatedly sample obstacles until we get one that is wider than target obstacle width
    i = 0
    w = 0
    while w < target_w and i < 50:
        i += 1
        if i == 50:
            print("Max iterations reached!")
        obj, obj_background, h, w = get_obj(obj_dict, obj_cat)
    target_h = int(h * target_w / w)

    # If image is still too large, make sure we do not upsample
    if target_w > w:
        target_w = w
        target_h = int(h * target_w / w)
    if target_h > h:
        target_h = h
        target_w = int(w * target_h / h)
    # Resize
    if target_h != h:
        obj = cv2.resize(obj, dsize=(target_w, target_h))

    # Get final position, including an x offset
    x_offset = random.randrange(-int(target_w / 2), int(target_w / 2))
    left = int(midpoint[0] - target_w/2 + x_offset)
    top = int(midpoint[1] - target_h)

    # Create an overlay image with the object of the same size as the underlying image
    obj = cv2.copyMakeBorder(obj, top, img_h - top - target_h, left,
                             img_w - left - target_w, cv2.BORDER_CONSTANT,
                             value=(0, 0, 0, 0))
    alpha = obj[:, :, 3:].astype('float32') / 255.0
    obj = obj[:, :, :3]

    if np.unique(alpha).shape[0] == 2:
        # crop a bit from the border of the mask as it may not be perfect
        alpha = sp.ndimage.uniform_filter(alpha, size=3).astype('int32')
        # now we smooth the binary mask around the corners to make the overlay more
        # realistic
        my_uniform_filter_size = 3
        alpha = sp.ndimage.uniform_filter(alpha.astype('float32'), size=my_uniform_filter_size)
        alpha = sp.ndimage.uniform_filter(alpha, size=my_uniform_filter_size)

    binary_mask = (1 - alpha < 0.5)[..., 0].astype('int8')

    # usually the object does not have the same brightness as then underlying patch
    # we do some gamma and brightness correction here, gamma noise should only be applied to rgb
    # https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/,
    my_low_gamma = 0.9
    my_up_gamma = 1.1
    gamma = rand.uniform(my_low_gamma, my_up_gamma)
    # look up table
    lut = np.array([((i / 255.0) ** (1 / gamma)) * 255
                    for i in np.arange(0, 256)])
    obj = lut[obj]

    # correct brightness of the object towards the mean brightness of the overlayed pixels
    mean_obj = np.mean(obj[binary_mask == 1][::4], dtype='float32')
    mean_background = np.mean(
        base_blob['rgb'][binary_mask == 1][::4], dtype='float32')
    obj[binary_mask == 1] += (mean_background - mean_obj) / 2
    obj = np.maximum(obj, 0)
    obj = np.minimum(obj, 255)

    # add linear motion blur s.t. it is 0 at top/img_h = 0.5 --> max_blur at top/img_h = 0.8]
    max_blur = 50
    my_motion_blur_steps = max_blur/(0.8-0.5) * top/img_h - max_blur*0.5/(0.8-0.5)
    my_motion_blur_steps = 1 if my_motion_blur_steps < 1 else int(my_motion_blur_steps) # minimum is 1
    print(f"Top/img_h: {top/img_h} --> {my_motion_blur_steps} motion blur steps")
    obj = motion_blurred_image(obj, my_motion_blur_steps)
    alpha = motion_blurred_image(alpha, my_motion_blur_steps)
    alpha = np.expand_dims(alpha, -1)
    # ### TODO: remove
    # output_rgb = cv2.cvtColor(obj.astype(np.uint8), cv2.COLOR_BGR2RGB)
    # output_image_pil = Image.fromarray(output_rgb)
    # output_path = os.path.join(OUTPUT_PATH, f"aaa_motion_blur.png")
    # output_image_pil.save(output_path, mode='RGB')

    # adapt color histogram
    # obj = histogram_adaptation(obj.astype('int'), obj_background.astype('int')) # TODO: evaluate if this makes sense

    # add depth blur
    # my_depth_blur
    a = 3
    b = 2
    c = 1
    d = 1

    if img_h - top - target_h > img_h / 2 and left > img_w / 4 and target_w < img_w / 2:
        # object is far along the road, extra blur
        obj = sp.ndimage.uniform_filter(obj, size=[a, a, b])
    else:
        obj = sp.ndimage.uniform_filter(obj, size=[c, c, d])

    # add color noise
    my_mean = 0
    my_std = 1
    noise = rand.normal(my_mean, my_std, obj.shape)
    obj = (obj + noise).astype('int32')

    # add glow effect
    my_weight = 0
    my_steps = 5
    my_sigma = [5, 5, 1]
    obj = glow_effect(obj, my_weight, my_steps, my_sigma)

    # map all labels on the overlay to ignored class 2
    labels[alpha[..., 0] > 0.01] = 2

    # all ignored labels (that are i.e. also not part of training) should be ignored
    # in the mask as well
    # binary_mask[np.logical_and(labels == -1, binary_mask == 0)] = -1

    blob = {
        # mixing by alpha channel
        'rgb': (alpha * obj[:, :, :3] +
                (1 - alpha) * base_blob['rgb']).astype('uint8'),
        'mask': labels.astype('int32'),
        'orig': base_blob['rgb'].astype(np.uint8),
        'labels': labels.astype('int32')}
    return blob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_images',
                            type=int,
                            default=20,
                            help='maximum number of images to be processed')
    parser.add_argument('--start_image_name',
                            type=str,
                            default="rs07500",
                            help='name of start image')
    parser.add_argument('--end_image_name',
                            type=str,
                            default="rs08499",
                            help='name of end image')
    parser.add_argument('--max_obstacles',
                            type=int,
                            default=100,
                            help='maximum number of obstacles to be processed')
    parser.add_argument('--crop_width',
                            type=int,
                            default=224,
                            help='width of the visualization_image crops in pixels')
    parser.add_argument('--crop_aspect_ratio',
                            type=float,
                            default=1.0,
                            help='aspect ratio of the visualization_image crops')
    parser.add_argument('--crop_distance_factor',
                            type=float,
                            default=2,
                            help='factor between crop width and projected distance between rails on bottom')
    parser.add_argument('--main_rail_only',
                            type=int,
                            default=1,
                            help='whether or not to extract images only from main rails')
    parser.add_argument('--min_distance',
                            type=int,
                            default=50,
                            help='minimum distance between rails to place obstacle there')
    parser.add_argument('--output_path',
                            type=str,
                            default="/media/matthias/sandisk/datasets/FishyscapesFull1",
                            help='output path')
    parser.add_argument('--input_path_rs19',
                            type=str,
                            default="/media/matthias/sandisk/datasets/rs19_val",
                            help='input path to rs19')
    parser.add_argument('--input_path_voc',
                            type=str,
                            default="/media/matthias/sandisk/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012",
                            help='input path to pascal voc')
    args = parser.parse_args()


    OBSTACLE_IMAGE_PATH = os.path.join(args.input_path_voc, "JPEGImages")  # jpg
    OBSTACLE_IMAGE_EXTENSION = ".jpg"
    OBSTACLE_OBJ_SEG_PATH = os.path.join(args.input_path_voc, "SegmentationObject/")  # png
    OBSTACLE_CLASS_SEG_PATH = os.path.join(args.input_path_voc, "SegmentationClass/")  # png
    OBSTACLE_SEG_EXTENSION = ".png"
    OBSTACLE_XML_PATH = os.path.join(args.input_path_voc, "Annotations/")  # png
    OBSTACLE_XML_EXTENSION = ".xml"
    OBSTACLE_CLASSES = [2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20]
    OBSTACLE_WIDTH_METER = [0, 5, 2, 1, 5, 0.5, 5, 2, 0.5, 1, 2, 1.5, 1, 2, 1.5, 1, 0.5, 1.5, 2, 5, 1]
    SAMPLING_LIST = ["humans", "humans", "animals", "animals", "objects", "vehicles"]
    SAMPLING_DICT = {"humans": [15], "animals": [8, 10, 12, 13, 17], "objects": [9, 11, 18, 20], "vehicles": [2, 6, 7, 14, 19]}
    #0:   background
    #1:   aeroplane
    #2:   bicycle
    #3:   bird
    #4:   boat
    #5:   bottle
    #6:   bus
    #7:   car
    #8:   cat
    #9:   chair
    #10:  cow
    #11:  diningtable
    #12:  dog
    #13:  horse
    #14:  motorbike
    #15:  person
    #16:  pottedplant
    #17:  sheep
    #18:  sofa
    #19:  train
    #20:  tvmonitor

    RAIL_IMAGE_PATH = os.path.join(args.input_path_rs19, "jpgs/rs19_val") #jpg
    RAIL_IMAGE_EXTENSION = ".jpg"
    RAIL_SEG_PATH = os.path.join(args.input_path_rs19, "uint8/rs19_val")
    RAIL_SEG_EXTENSION = ".png"
    JSON_PATH = os.path.join(args.input_path_rs19, "jsons/rs19_val")  # json
    JSON_EXTENSION = ".json"
    OUTPUT_PATH = os.path.join(args.output_path, "fishy")
    OUTPUT_ORIG_PATH = os.path.join(args.output_path, "orig")
    OUTPUT_MASK_PATH = os.path.join(args.output_path, "masks_fishy")
    OUTPUT_ORIG_MASK_PATH = os.path.join(args.output_path, "masks_orig")
    OUTPUT_OBSTACLE_PATH = os.path.join(args.output_path, "obstacles")
    OUTPUT_JSON_PATH = os.path.join(args.output_path, "jsons")
    SEED = 1
    random.seed(SEED)

    main(args)
