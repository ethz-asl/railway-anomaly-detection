import os
import numpy as np
import argparse
import h5py
import torch
import random
import presets
from autoencoder_networks import AeSegParam02
from PIL import Image
import io

class RailSem19CroppedDatasetSegmentation(torch.utils.data.Dataset):
    def __init__(self, data_path, transforms, mode="train", train_fraction=0.9):
        self.root = data_path
        self.transforms = transforms
        self.data_path = data_path
        with h5py.File(data_path, 'r') as hdf:
            G_images = hdf.get('images')
            self.images = sorted(G_images.keys())
            G_masks = hdf.get('masks')
            self.masks = sorted(G_masks.keys())

            # load all image files, sorting them to ensure that they are aligned
            #self.images = [image for image in sorted(os.listdir(os.path.join(self.root, "images"))) if image != ".gitignore"]
            #self.masks = [mask for mask in sorted(os.listdir(os.path.join(self.root, "masks"))) if mask != ".gitignore"]
            if len(self.masks) != len(self.images):
                print(f"ATTENTION: {len(self.images)} images but {len(self.masks)} masks!")
            # Train validation split
            train_length = int(len(self.images) * train_fraction)
            if mode == "train":
                self.images = self.images[:train_length]
                self.masks = self.masks[:train_length]
            elif mode == "val":
                self.images = self.images[train_length:]
                self.masks = self.masks[train_length:]
            else:
                mode = "full"  # take all images

            self.image_data = dict()
            self.mask_data = dict()
            # for image_name, mask_name in zip(self.images, self.masks):
            #     self.image_data[image_name] = np.array(G_images.get(image_name))
            #     self.mask_data[mask_name] = np.array(G_masks.get(mask_name))
        print(f"Segmentation Dataset {mode}: {len(self.images)} images from {self.root}")

    def __getitem__(self, idx):
        # load images and masks
        # img_path = os.path.join(self.root, "images", self.images[idx])
        # mask_path = os.path.join(self.root, "masks", self.masks[idx])
        # img = Image.open(img_path).convert("RGB")
        # target = Image.open(mask_path) # no need to convert masks to RGB

        # img = self.image_data[self.images[idx]]
        # target = self.mask_data[self.masks[idx]]
        with h5py.File(self.data_path, 'r') as hdf:
            G_images = hdf.get('images')
            G_masks = hdf.get('masks')
            img = np.array(G_images.get(self.images[idx]))
            target = np.array(G_masks.get(self.masks[idx]))

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

def main(args):

    # Go over all input images
    with h5py.File(args.input_path, 'r') as hdf:
        G_images_in1 = hdf.get('images')
        images = sorted(G_images_in1.keys())
        G_masks_in1 = hdf.get('masks')
        masks = sorted(G_masks_in1.keys())
        if len(masks) != len(images):
            print(f"ATTENTION: {len(images)} images but {len(masks)} masks!")
            return

    with h5py.File(args.output_path, 'w') as hdf:
        G_images = hdf.create_group('images')
        G_masks = hdf.create_group('masks')
        G_gan = hdf.create_group('gan')

        # Random seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.set_flush_denormal(True)
        transforms = presets.SegmentationPresetEval(base_size=224, crop=False)

        # Load model
        if args.ae_type == "AeSegParam02_8110":
            model = AeSegParam02(c_seg=8, c_ae=1, c_param=1, mode="none", ratio=args.color_space_ratio, act=args.g_act)
        elif args.ae_type == "AeSegParam02_8210":
            model = AeSegParam02(c_seg=8, c_ae=2, c_param=1, mode="none", ratio=args.color_space_ratio, act=args.g_act)
        elif args.ae_type == "AeSegParam0_8410":
            model = AeSegParam02(c_seg=8, c_ae=4, c_param=1, mode="none", ratio=args.color_space_ratio, act=args.g_act)
        elif args.ae_type == "AeSegParam02_8810":
            model = AeSegParam02(c_seg=8, c_ae=8, c_param=1, mode="none", ratio=args.color_space_ratio, act=args.g_act)
        elif args.ae_type == "AeSegParam02_8111":
            model = AeSegParam02(c_seg=8, c_ae=1, c_param=1, mode="remap", ratio=args.color_space_ratio, act=args.g_act)
        elif args.ae_type == "AeSegParam02_8211":
            model = AeSegParam02(c_seg=8, c_ae=2, c_param=1, mode="remap", ratio=args.color_space_ratio, act=args.g_act)
        elif args.ae_type == "AeSegParam02_8411":
            model = AeSegParam02(c_seg=8, c_ae=4, c_param=1, mode="remap", ratio=args.color_space_ratio, act=args.g_act)
        elif args.ae_type == "AeSegParam02_8811":
            model = AeSegParam02(c_seg=8, c_ae=8, c_param=1, mode="remap", ratio=args.color_space_ratio, act=args.g_act)
        elif args.ae_type == "AeSegParam02_8112":
            model = AeSegParam02(c_seg=8, c_ae=1, c_param=1, mode="zero", ratio=args.color_space_ratio, act=args.g_act)
        elif args.ae_type == "AeSegParam02_8212":
            model = AeSegParam02(c_seg=8, c_ae=2, c_param=1, mode="zero", ratio=args.color_space_ratio, act=args.g_act)
        elif args.ae_type == "AeSegParam02_8412":
            model = AeSegParam02(c_seg=8, c_ae=4, c_param=1, mode="zero", ratio=args.color_space_ratio, act=args.g_act)
        elif args.ae_type == "AeSegParam02_8812":
            model = AeSegParam02(c_seg=8, c_ae=8, c_param=1, mode="zero", ratio=args.color_space_ratio, act=args.g_act)

        else:
            print(f"No such autoencoder type: {args.ae_type}!")
            return -1

        device = torch.device(args.device)
        model.to(device)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        model.eval()

        # Inference
        with torch.no_grad():
            for idx in range(len(images)):
                print(f"Image {idx}/{len(images)} ...")
                # Read image
                with h5py.File(args.input_path, 'r') as hdf_in:
                    G_images_in = hdf_in.get('images')
                    G_masks_in = hdf_in.get('masks')
                    image_bin_np = np.array(G_images_in.get(images[idx]))
                    mask_bin_np = np.array(G_masks_in.get(masks[idx]))
                # Read + write image
                # image_path = images[idx]
                # image_name = os.path.splitext(os.path.basename(image_path))[0]
                # with open(image_path, 'rb') as image_f:
                #     image_file = image_f.read()
                # image_bin_np = np.asarray(image_file)
                image_pil = Image.open(io.BytesIO(image_bin_np))
                img = np.array(image_pil)
                G_images.create_dataset(images[idx], data=image_bin_np)

                # Read + write mask
                # mask_path = masks[idx]
                # mask_name = image_name
                # with open(mask_path, 'rb') as mask_f:
                #     mask_file = mask_f.read()
                # mask_bin_np = np.asarray(mask_file)
                mask_pil = Image.open(io.BytesIO(mask_bin_np))
                target = np.array(mask_pil)
                G_masks.create_dataset(masks[idx], data=mask_bin_np)

                # Transform to torch
                if transforms is not None:
                    img, target = transforms(img, target)
                img, target = img.to(device), target.to(device)
                img = img.unsqueeze(dim=0)

                # Inference
                outputs = model(img)
                output_ae = outputs["out_aa"]

                # Re-convert
                if args.g_act == "tanh":
                    output_aa_re_pil, _ = presets.re_convert_tanh(output_ae, output_ae)  # no de-normalization
                else:
                    output_aa_re_pil, _ = presets.re_convert(output_ae, output_ae)

                # Save image in hdf5
                cache_path = os.path.join(args.cache_path, "cache_gan.png")
                output_aa_re_pil.save(cache_path)

                with open(cache_path, 'rb') as image_f:
                    image_file = image_f.read()
                image_bin_np = np.asarray(image_file)
                G_gan.create_dataset(images[idx], data=image_bin_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_path',
                        type=str,
                        default="/tmp/",
                        help='path folder for intermediate storage')
    parser.add_argument('--input_path',
                        type=str,
                        default="/path/to/Railsem19Croppedv1.h5",
                        help='path to the input .h5 file')
    parser.add_argument('--output_path',
                        type=str,
                        default="Railsem19CroppedAEv1.h5",
                        help='path to the output .h5 file')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='seed')
    parser.add_argument("--ae_type", default="AeSegParam02_8810", type=str, help="which Autoencoder")
    parser.add_argument("--device", default="cpu", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--checkpoint", default="./trained_models/ae_model_199.pth", type=str,
                        help="path of checkpoint")
    parser.add_argument("--g_act", default="tanh", type=str, help="generator activation")
    parser.add_argument("--color_space_ratio", default=0.1, type=float, help="color space ratio for each channel")

    args = parser.parse_args()
    main(args)