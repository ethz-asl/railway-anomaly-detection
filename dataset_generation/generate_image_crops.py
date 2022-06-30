import sys
import os
import cv2
import numpy as np
import glob
import json
import argparse
from utils import compute_midpoints


def main(args):
    # Prepare output directories
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    if not os.path.exists(OUTPUT_MASK_PATH):
        os.makedirs(OUTPUT_MASK_PATH)
    if not os.path.exists(VISUALIZATION_PATH):
        os.makedirs(VISUALIZATION_PATH)
    if MODE == "fishyrails":
        if not os.path.exists(OUTPUT_ORIG_PATH):
            os.makedirs(OUTPUT_ORIG_PATH)
        if not os.path.exists(OUTPUT_ORIG_MASK_PATH):
            os.makedirs(OUTPUT_ORIG_MASK_PATH)

    # Get list of images and segmentation masks and check if everything is alright
    image_regex = os.path.join(IMAGE_PATH, f"*{IMAGE_EXTENSION}")
    image_list = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(image_regex)]
    if MODE == "rs19":
        image_list = [x for x in image_list if args.start_image_name <= x <= args.end_image_name]
    image_list.sort()
    seg_regex = os.path.join(SEG_PATH, f"*{SEG_EXTENSION}")
    seg_list = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(seg_regex)]
    if MODE == "rs19":
        seg_list = [x for x in seg_list if args.start_image_name <= x <= args.end_image_name]
    seg_list.sort()
    json_regex = os.path.join(JSON_PATH, f"*{JSON_EXTENSION}")
    json_list = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(json_regex)]
    if MODE == "rs19":
        json_list = [x for x in json_list if args.start_image_name <= x <= args.end_image_name]
    json_list.sort()
    print(f"Found {len(image_list)} images from {IMAGE_PATH})")
    print(f"Found {len(seg_list)} segmentations from {SEG_PATH})")
    print(f"Found {len(seg_list)} jsons from {JSON_PATH})")
    if len(image_list) != len(seg_list):
        print("Number of images and segmentations does not match!")
        return -1
    with open(DENSE_LABELS_PATH) as config_file:
        dense_labels = json.load(config_file)

    # Iterate over images
    counter = 0
    for image_name in image_list:
        print(f"Processing visualization_image {counter}")
        # Load visualization_image data (RGB, segmentation, annotations)
        image_path = os.path.join(IMAGE_PATH, f"{image_name}{IMAGE_EXTENSION}")
        seg_path = os.path.join(SEG_PATH, f"{image_name}{SEG_EXTENSION}")
        json_path = os.path.join(JSON_PATH, f"{image_name}{JSON_EXTENSION}")
        image = cv2.imread(image_path)
        seg = cv2.imread(seg_path, 0)  # grayscale mode
        if MODE == "fishyrails":
            image_orig_path = os.path.join(IMAGE_ORIG_PATH, f"{image_name}{IMAGE_ORIG_EXTENSION}")
            seg_orig_path = os.path.join(SEG_ORIG_PATH, f"{image_name}{SEG_ORIG_EXTENSION}")
            image_orig = cv2.imread(image_orig_path)
            seg_orig = cv2.imread(seg_orig_path, 0)  # grayscale mode
        with open(json_path) as json_file:
            json_data = json.load(json_file)
        # Create image crops from visualization_image
        visualization_image, cropped_images, cropped_segmentations, cropped_orig_images, cropped_orig_segmentations, frame \
            = create_image_crops(args, json_data, image, seg, image_orig, seg_orig)
        # Store visualization and visualization_image crops
        cv2.imwrite(os.path.join(VISUALIZATION_PATH, f"visualization_{image_name}.png"), visualization_image)
        for idx, crop in enumerate(cropped_images):
            cv2.imwrite(os.path.join(OUTPUT_PATH, f"{image_name}_{idx:02}.png"), crop)
        for idx, crop in enumerate(cropped_segmentations):
            cv2.imwrite(os.path.join(OUTPUT_MASK_PATH, f"{image_name}_{idx:02}_mask.png"), crop)

        if MODE == "fishyrails":
            for idx, crop in enumerate(cropped_orig_images):
                cv2.imwrite(os.path.join(OUTPUT_ORIG_PATH, f"{image_name}_{idx:02}.png"), crop)
            for idx, crop in enumerate(cropped_orig_segmentations):
                cv2.imwrite(os.path.join(OUTPUT_ORIG_MASK_PATH, f"{image_name}_{idx:02}.png"), crop)
        counter += 1
        if counter > args.max_images:
            return 0


def create_image_crops(args, input_json, input_image, input_segmentation, input_orig_image=None, input_orig_segmentation=None):
    # Create visualization visualization_image copy and empty cropped images list
    visualization_image = input_image.copy()
    cropped_images = list()
    cropped_segmentations = list()
    cropped_orig_images = list() if MODE == "fishyrails" else None
    cropped_orig_segmentations = list() if MODE == "fishyrails" else None

    # Extract area between rails
    if MODE == "rs19":
        segmentation_mask = np.isin(input_segmentation, [12, 17, 18, 3]).astype(np.uint8) # rail-track, rail non-drivable, rail drivable
    elif MODE == "fishyrails":
        segmentation_mask = input_segmentation.astype(np.uint8)
    segmentation_mask_extended = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2)
    visualization_image[segmentation_mask_extended==1] = 200 # color visualization image
    # segmentation_mask = segmentation_mask.astype(np.uint8)
    # If there are no real rails in visualization_image (tram rails do not count), ignore visualization_image
    if np.count_nonzero(np.isin(input_segmentation, [12])) == 0 and MODE == "rs19":
        return input_image, cropped_images, cropped_segmentations, input_json["frame"]

    # Extract rails
    for obj in input_json["objects"]:
        if "polyline-pair" in obj:
            # Extract all available rails (modifies visualization visualization_image)
            draw_rails(visualization_image, obj["polyline-pair"])
            # Crop images from main rails (modifies visualization visualization_image and appends to cropped_images)
            crop_images_from_rails(visualization_image, input_image, segmentation_mask, input_orig_image, input_orig_segmentation, cropped_images,
                                   cropped_segmentations, cropped_orig_images, cropped_orig_segmentations,obj["polyline-pair"],
                                   crop_distance_factor=args.crop_distance_factor, crop_width=args.crop_width,
                                   output_width=args.output_width, crop_aspect_ratio=args.crop_aspect_ratio)
    return visualization_image, cropped_images, cropped_segmentations, cropped_orig_images, cropped_orig_segmentations, input_json["frame"]


def draw_rails(visualization_image, coords):
    # Draw rails on visualization image
    col = (255, 255, 0)
    # Left rail of a rail pair has index 0, right rail of a rail pair has index 1
    rails_draw = [np.around(np.array(coords[i])).astype(np.int32) for i in range(2)]
    cv2.polylines(visualization_image, rails_draw, False, col)


def crop_images_from_rails(visualization_image, raw_image, segmentation_mask, orig_image, orig_mask, cropped_images, cropped_segmentations,
                           cropped_orig_images, cropped_orig_segmentations,
                           coords, crop_distance_factor=2, crop_width=128, output_width=224, crop_aspect_ratio=1):
    """
    Function crops images from rails such that the image width is crop_distance_factor * distance between the rails
    for different points in the rails.
    :param visualization_image: input visualization image
    :param raw_image: raw input image
    :param cropped_images: list of image crops per image, will be appended to
    :param coords: coordinates of polygons for left rail (coords[0]) and right rail (coords[1])
    :param crop_distance_factor: factor between crop width and rail distance
    :param crop_width: minimum width of rectangular crops
    :param output_width: width of final cropped image, upsampled from crop_width
    :param crop_aspect_ratio: aspect ratio of final cropped image
    :return: Nothing, modifies visualization_image and appends to cropped_images list
    """
    height, width, _ = raw_image.shape
    col = (255, 0, 255)
    connection_col = (0, 0, 0)

    midpoints, distances, short, long, _ = compute_midpoints(coords)

    # If desired: Discard rails too far from the lower image center (only keep main rails)
    if args.main_rail_only:
        is_main_rail = False
    else:
        # all rails are main rails
        is_main_rail = True
    for midpoint in midpoints:
        if abs(midpoint[0] - width/2) < width / 6 and height - midpoint[1] < height / 10:
            is_main_rail = True
    if not is_main_rail:
        return

    # Visualization: Print connections between corresponding points
    for short, long in zip(short, long):
        cv2.line(visualization_image, tuple(short), tuple(long), connection_col)

    # Visualization: Print midpoint line
    midpoint_line = [np.around(np.array(midpoints)).astype(np.int32)]
    cv2.polylines(visualization_image, midpoint_line, False, col)

    # Visualization: overlay segmentation mask
    # segmentation_mask_extended = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2).astype(bool)
    # visualization_image[segmentation_mask_extended] = 50
    # visualization_image = cv2.addWeighted(segmentation_mask_extended, 1, visualization_image, 0, 0)

    # Extract crops from image and visualize them as rectangles
    box_col = (255, 255, 255)
    prev_midpoint = (99999, 99999)
    up_shift = args.y_midpoint_location
    for midpoint, distance in zip(midpoints, distances):
        # Ensure that midpoints of crops are at least "half a crop" away from each other
        midpoint_distance = np.sqrt((prev_midpoint[0] - midpoint[0]) ** 2 + (prev_midpoint[1] - midpoint[1]) ** 2)
        actual_width = crop_distance_factor*distance
        if midpoint_distance > actual_width/2 and actual_width >= crop_width:
            top_left = (int(midpoint[0] - actual_width / 2), int(midpoint[1] - (1 - up_shift) * actual_width / crop_aspect_ratio))
            bottom_right = (int(midpoint[0] + actual_width / 2), int(midpoint[1] + up_shift * actual_width / crop_aspect_ratio))
            # make sure that crop is completely inside image
            if top_left[1] >= 0 and top_left[0] >= 0 and bottom_right[1] < height and bottom_right[0] < width:
                crop = raw_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]  # np: [height, width, 3]
                cv2.rectangle(visualization_image, top_left, bottom_right, box_col, 2)
                crop = cv2.resize(crop, (output_width, int(output_width / crop_aspect_ratio)))
                cropped_images.append(crop)
                segmentation_crop = segmentation_mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                segmentation_crop = cv2.resize(segmentation_crop, (output_width, int(output_width / crop_aspect_ratio)),
                                               interpolation=cv2.INTER_NEAREST)
                cropped_segmentations.append(segmentation_crop)
                if orig_image is not None:
                    orig_image_crop = orig_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                    orig_image_crop = cv2.resize(orig_image_crop, (output_width, int(output_width / crop_aspect_ratio)))
                    cropped_orig_images.append(orig_image_crop)
                if orig_mask is not None:
                    orig_mask_crop = orig_mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                    orig_mask_crop = cv2.resize(orig_mask_crop,
                                                   (output_width, int(output_width / crop_aspect_ratio)),
                                                   interpolation=cv2.INTER_NEAREST)
                    cropped_orig_segmentations.append(orig_mask_crop)
                prev_midpoint = midpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_images',
                        type=int,
                        default=20,
                        help='maximum number of images to be processed')
    parser.add_argument('--start_image_name',
                        type=str,
                        default="rs00000",
                        help='name of start image')
    parser.add_argument('--end_image_name',
                        type=str,
                        default="rs07499",
                        help='name of end image')
    parser.add_argument('--crop_width',
                        type=int,
                        default=128,
                        help='width of the visualization_image crops in pixels')
    parser.add_argument('--output_width',
                        type=int,
                        default=224,
                        help='image output width. Up/downsampled from crop width.')
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
    parser.add_argument('--y_midpoint_location',
                        type=float,
                        default=0.25,
                        help='y location of the midpoint in the image crop as a fraction of the crop height. '  
                             '0 means midpoint is at the bottom of the crop, 1 means midpoint is on top, '
                             '0.5 means midpoint is in middle of crop')
    parser.add_argument('--mode',
                        type=str,
                        default="rs19",
                        help='fishyrails or rs19')
    parser.add_argument('--output_path',
                        type=str,
                        default="/path/to/Railsem19Croppedv1",
                        help='fishyrails or rs19')
    parser.add_argument('--input_path',
                        type=str,
                        default="/path/to/rs19_val",
                        help='rs19_val or fishyrails directory')
    args = parser.parse_args()
    MODE = "fishyrails"  # "rs19" or "fishyrails"

    if MODE == "rs19":
        IMAGE_PATH = os.path.join(args.input_path, "jpgs/rs19_val")  # jpg
        IMAGE_EXTENSION = ".jpg"
        SEG_PATH = os.path.join(args.input_path, "uint8/rs19_val")  # png
        SEG_EXTENSION = ".png"
        JSON_PATH = os.path.join(args.input_path, "jsons/rs19_val")  # json
        JSON_EXTENSION = ".json"
        OUTPUT_PATH = os.path.join(args.output_path, "images")
        OUTPUT_MASK_PATH = os.path.join(args.output_path, "masks")
        VISUALIZATION_PATH = os.path.join(args.output_path, "visualizations")
        DENSE_LABELS_PATH = os.path.join(os.getcwd(), "rs19_val/rs19-config.json")
    elif MODE == "fishyrails":
        IMAGE_PATH = os.path.join(args.input_path, "fishy")  # png
        IMAGE_EXTENSION = ".png"
        SEG_PATH = os.path.join(args.input_path, "masks_fishy")  # png
        SEG_EXTENSION = ".png"
        IMAGE_ORIG_PATH = os.path.join(args.input_path, "orig")  # png
        IMAGE_ORIG_EXTENSION = ".png"
        SEG_ORIG_PATH = os.path.join(args.input_path, "masks_orig")  # png
        SEG_ORIG_EXTENSION = ".png"
        JSON_PATH = os.path.join(args.input_path, "jsons")  # json
        JSON_EXTENSION = ".json"
        OUTPUT_PATH = os.path.join(args.output_path, "images_fishy")
        OUTPUT_MASK_PATH = os.path.join(args.output_path, "masks_fishy")
        OUTPUT_ORIG_PATH = os.path.join(args.output_path, "images_orig")
        OUTPUT_ORIG_MASK_PATH = os.path.join(args.output_path, "masks_orig")
        VISUALIZATION_PATH = os.path.join(args.output_path, "visualizations")
        DENSE_LABELS_PATH = os.path.join(os.getcwd(), "rs19_val/rs19-config.json")
    main(args)
