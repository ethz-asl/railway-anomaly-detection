import os
import numpy as np
import argparse
import h5py
from PIL import Image

def main(args):
    # load all image files, sorting them to ensure that they are aligned
    im_fishy_path = os.path.join(args.input_path, "images_fishy")
    images_fishy = [os.path.join(im_fishy_path, image) for image in sorted(os.listdir(im_fishy_path))]
    im_orig_path = os.path.join(args.input_path, "images_orig")
    images_orig = [os.path.join(im_orig_path, image) for image in sorted(os.listdir(im_orig_path))]
    mask_fishy_path = os.path.join(args.input_path, "masks_fishy")
    masks_fishy = [os.path.join(mask_fishy_path, mask) for mask in sorted(os.listdir(mask_fishy_path))]
    mask_orig_path = os.path.join(args.input_path, "masks_orig")
    masks_orig = [os.path.join(mask_orig_path, mask) for mask in sorted(os.listdir(mask_orig_path))]
    if len(masks_fishy) != len(images_fishy):
        print(f"ATTENTION: {len(images_fishy)} fishy images but {len(masks_fishy)} fishy masks!")
        return
    with h5py.File(os.path.join(args.input_path, f"{args.output_name}.h5"), 'w') as hdf:
        # Fishy Images
        G_images_fishy = hdf.create_group('images_fishy')
        for idx, image_path in enumerate(images_fishy):
            print(f"Processing Fishy Image {idx}/{len(images_fishy)}")
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            with open(image_path, 'rb') as image_f:
                image_file = image_f.read()
            image_np = np.asarray(image_file)
            G_images_fishy.create_dataset(image_name, data=image_np)
        # Orig Images
        G_images_orig = hdf.create_group('images_orig')
        for idx, image_path in enumerate(images_orig):
            print(f"Processing Orig Image {idx}/{len(images_orig)}")
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            with open(image_path, 'rb') as image_f:
                image_file = image_f.read()
            image_np = np.asarray(image_file)
            G_images_orig.create_dataset(image_name, data=image_np)
        # Fishy Masks
        G_masks_fishy = hdf.create_group('masks_fishy')
        for idx, mask_path in enumerate(masks_fishy):
            print(f"Processing Fishy Mask {idx}/{len(masks_fishy)}")
            mask_name = os.path.splitext(os.path.basename(mask_path))[0]
            with open(mask_path, 'rb') as mask_f:
                mask_file = mask_f.read()
            mask_np = np.asarray(mask_file)
            G_masks_fishy.create_dataset(mask_name, data=mask_np)
        # Orig Masks
        G_masks_orig = hdf.create_group('masks_orig')
        for idx, mask_path in enumerate(masks_orig):
            print(f"Processing Orig Mask {idx}/{len(masks_orig)}")
            mask_name = os.path.splitext(os.path.basename(mask_path))[0]
            with open(mask_path, 'rb') as mask_f:
                mask_file = mask_f.read()
            mask_np = np.asarray(mask_file)
            G_masks_orig.create_dataset(mask_name, data=mask_np)
    print(f"Saved to {os.path.join(args.input_path, f'{args.output_name}.h5')}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
                        type=str,
                        default="/path/to/FishyrailsCroppedv1",
                        help='path to the directory structure to be stored as hdf5 file')
    parser.add_argument('--output_name',
                        type=str,
                        default="FishyrailsCroppedv1",
                        help='name of output hdf5 file')
    args = parser.parse_args()
    main(args)
