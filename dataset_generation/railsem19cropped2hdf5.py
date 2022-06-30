import os
import numpy as np
import argparse
import h5py
from PIL import Image

def main(args):
    # load all image files, sorting them to ensure that they are aligned
    im_path = os.path.join(args.input_path, "images")
    images = [os.path.join(im_path, image) for image in sorted(os.listdir(im_path))]
    mask_path = os.path.join(args.input_path, "masks")
    masks = [os.path.join(mask_path, mask) for mask in sorted(os.listdir(mask_path))]
    if len(masks) != len(images):
        print(f"ATTENTION: {len(images)} images but {len(masks)} masks!")
        return
    if args.with_params == 1:
        param_path = os.path.join(args.input_path, "params")
        params = [os.path.join(param_path, param) for param in sorted(os.listdir(param_path))]
        if len(params) != len(images):
            print(f"ATTENTION: {len(images)} images but {len(params)} params!")
            return


    with h5py.File(os.path.join(args.input_path, f"{args.output_name}.h5"), 'w') as hdf:
        # Images
        G_images = hdf.create_group('images')
        for idx, image_path in enumerate(images):
            print(f"Processing Image {idx}/{len(images)}")
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            with open(image_path, 'rb') as image_f:
                image_file = image_f.read()
            image_np = np.asarray(image_file)
            G_images.create_dataset(image_name, data=image_np)
        # Masks
        G_masks = hdf.create_group('masks')
        for idx, mask_path in enumerate(masks):
            print(f"Processing Mask {idx}/{len(masks)}")
            mask_name = os.path.splitext(os.path.basename(mask_path))[0]
            with open(mask_path, 'rb') as mask_f:
                mask_file = mask_f.read()
            mask_np = np.asarray(mask_file)
            G_masks.create_dataset(mask_name, data=mask_np)
        # Params
        if args.with_params == 1:
            G_params = hdf.create_group('params')
            for idx, param_path in enumerate(params):
                print(f"Processing Param {idx}/{len(params)}")
                param_name = os.path.splitext(os.path.basename(param_path))[0]
                with open(param_path, 'rb') as param_f:
                    param_file = param_f.read()
                param_np = np.asarray(param_file)
                G_params.create_dataset(param_name, data=param_np)
    print(f"Saved to {os.path.join(args.input_path, f'{args.output_name}.h5')}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
                        type=str,
                        default="/path/to/Railsem19Croppedv1",
                        help='path to the directory structure to be stored as hdf5 file')
    parser.add_argument('--output_name',
                        type=str,
                        default="Railsem19Croppedv1",
                        help='name of output hdf5 file')
    parser.add_argument('--with_params',
                        type=int,
                        default=0,
                        help='with params or not')
    args = parser.parse_args()
    main(args)
