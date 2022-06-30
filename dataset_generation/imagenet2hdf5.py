import os
import numpy as np
import argparse
import h5py
from PIL import Image
import io

def main(args):
    # load all image files, sorting them to ensure that they are aligned
    sub_folders = list()
    images = list()
    for subfolder in sorted(os.listdir(args.input_path)):
        subfolder_path = os.path.join(args.input_path, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Processing subfolder {subfolder}")
            for image in sorted(os.listdir(subfolder_path)):
                image_path = os.path.join(subfolder_path, image)
                if image.endswith(".JPEG"):
                    images.append(image_path)
    print(f"Found {len(images)} images!")

    with h5py.File(os.path.join(args.input_path, f"{args.output_name}.h5"), 'w') as hdf:
        # Images
        G_images = hdf.create_group('images')
        for idx, image_path in enumerate(images):
            print(f"Processing Image {idx}/{len(images)}")
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            # image_np = np.array(Image.open(image_path))
            # Alternative: from brilands comment: https://github.com/h5py/h5py/issues/745
            # read back in with:
            # dset_read = f.get('binary_data')
            # dset_read_np = np.array(dset_read)
            # img_res = Image.open(io.BytesIO(dset_read_np))
            # img_res.show()
            with open(image_path, 'rb') as img_f:
                image_file = img_f.read()
            image_np = np.asarray(image_file)
            G_images.create_dataset(image_name, data=image_np)
    print(f"Saved to {os.path.join(args.input_path, f'{args.output_name}.h5')}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
                        type=str,
                        default="/path/to/ImageNet",
                        help='path to the directory structure to be stored as hdf5 file')
    parser.add_argument('--output_name',
                        type=str,
                        default="ImageNet",
                        help='name of output hdf5 file')
    args = parser.parse_args()
    main(args)
