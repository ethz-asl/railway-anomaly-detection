# from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import io
import numpy as np
import torch
from PIL import Image
import h5py
from torchvision.transforms import functional as F
import random
import presets
import transforms
import os



class RailSem19CroppedDatasetPatchSegmentation(torch.utils.data.Dataset):
    def __init__(self, data_path, data_path_image_net, mode="train", train_fraction=0.9, use_gan=True, use_neighbors=True, imagenet_ratio=0.5, secimage_ratio=0.1, no_seg=False):
        self.root = data_path
        self.data_path = data_path
        self.data_path_image_net = data_path_image_net
        self.base_transform = presets.GanSegmentationPreset(base_size=224, flip=0, crop=0)
        self.in_transform = presets.GanSegmentationPreset(base_size=224, flip=0, crop=1)
        self.use_gan = use_gan
        self.use_neighbors = use_neighbors
        self.imagenet_ratio = imagenet_ratio
        self.secimage_ratio = secimage_ratio
        self.no_seg = no_seg
        if self.use_gan and self.use_neighbors:
            self.channels = 30
        elif self.use_gan and not self.use_neighbors:
            self.channels = 6
        elif not self.use_gan and self.use_neighbors:
            self.channels = 15
        else:
            self.channels = 3

        # Gan Dataset
        with h5py.File(data_path, 'r') as hdf:
            G_images = hdf.get('images')
            print("G images")
            print(G_images)
            self.images = sorted(G_images.keys())
            G_masks = hdf.get('masks')
            self.masks = sorted(G_masks.keys())
            G_gan = hdf.get('gan')
            self.gan_images = sorted(G_gan.keys())

        # Image Net dataset
        with h5py.File(data_path_image_net, 'r') as hdf:
            G_images_in = hdf.get('images')
            self.images_in = list(G_images_in.keys())
            random.shuffle(self.images_in)

        if len(self.masks) != len(self.images) or len(self.gan_images) != len(self.images):
            print(
                f"ATTENTION: {len(self.images)} images but {len(self.masks)} masks and {len(self.gan_images)} gan images!")

        # Train validation split
        train_length = int(len(self.images) * train_fraction)
        train_length_in = int(len(self.images_in) * train_fraction)
        if mode == "train":
            self.images = self.images[:train_length]
            self.gan_images = self.gan_images[:train_length]
            self.masks = self.masks[:train_length]
            self.images_in = self.images_in[:train_length_in]
        elif mode == "val":
            self.images = self.images[train_length:]
            self.gan_images = self.gan_images[train_length:]
            self.masks = self.masks[train_length:]
            self.images_in = self.images_in[train_length_in:]
        else:
            mode = "full"  # take all images

        self.in_length = len(self.images_in)
            # for image_name, mask_name in zip(self.images, self.masks):
            #     self.image_data[image_name] = np.array(G_images.get(image_name))
            #     self.mask_data[mask_name] = np.array(G_masks.get(mask_name))
        print(f"Segmentation Dataset {mode}: {len(self.images)} images from {self.root}")
        print(f"Segmentation Dataset {mode}: {self.in_length} images from {self.data_path_image_net}")

    def __getitem__(self, idx):
        img_idx = idx
        img_idx_in = random.randint(0, self.in_length - 1)
        img_idx_2 = img_idx
        while img_idx_2 == img_idx:
            img_idx_2 = random.randint(0, len(self.images) - 1)
        random_number_mode = random.random()
        if random_number_mode < self.imagenet_ratio:
            mode = 1
        elif random_number_mode > 1 - self.secimage_ratio:
            mode = 2
        else:
            mode = 0

        # load images and masks
        with h5py.File(self.data_path, 'r') as hdf:
            # image
            G_images = hdf.get('images')
            bytes_img = np.array(G_images.get(self.images[img_idx]))
            pil_image = Image.open(io.BytesIO(bytes_img))
            pil_image = pil_image.convert("RGB")
            img = np.array(pil_image)

            # gan image
            G_gan = hdf.get('gan')
            bytes_gan_img = np.array(G_gan.get(self.gan_images[img_idx]))
            pil_gan_image = Image.open(io.BytesIO(bytes_gan_img))
            pil_gan_image = pil_gan_image.convert("RGB")
            gan_img = np.array(pil_gan_image)

            # mask
            G_mask = hdf.get('masks')
            bytes_mask = np.array(G_mask.get(self.masks[img_idx]))
            pil_mask = Image.open(io.BytesIO(bytes_mask))
            pil_mask = pil_mask.convert("L")
            mask = np.array(pil_mask)

            # second image
            if mode == 2:
                bytes_img_2 = np.array(G_images.get(self.images[img_idx_2]))
                pil_image_2 = Image.open(io.BytesIO(bytes_img_2))
                pil_image_2 = pil_image_2.convert("RGB")
                img_2 = np.array(pil_image_2)

        with h5py.File(self.data_path_image_net, 'r') as hdf:
            G_images_in = hdf.get('images')
            bytes_img_in = np.array(G_images_in.get(self.images_in[img_idx_in]))
            pil_image_in = Image.open(io.BytesIO(bytes_img_in))
            # make sure image is RGB (necessary for grayscale or 4 channel images)
            pil_image_in = pil_image_in.convert("RGB")
            img_in = np.array(pil_image_in)
            H, W, C = img_in.shape
            assert C == 3, f"Channels != 3: Image {self.images_in[img_idx_in]} shape: {img_in.shape}"

        mask_zeros = np.zeros((224, 224), dtype=np.uint8)
        mask_ones = np.ones((H, W), dtype=np.uint8)
        img, mask = self.base_transform(img, mask)
        gan_img, mask_zeros = self.base_transform(gan_img, mask_zeros)
        img_in, mask_ones = self.in_transform(img_in, mask_ones)
        if mode == 2:
            mask_dummy = np.zeros((224, 224), dtype=np.uint8)
            img_2, _ = self.base_transform(img_2, mask_dummy)

        gan_img_r1 = torch.zeros_like(gan_img)
        gan_img_r1[:, :, :-27] = gan_img[:, :, 27:]
        gan_img_r2 = torch.zeros_like(gan_img)
        gan_img_r2[:, :, :-54] = gan_img[:, :, 54:]
        gan_img_l1 = torch.zeros_like(gan_img)
        gan_img_l1[:, :, 27:] = gan_img[:, :, :-27]
        gan_img_l2 = torch.zeros_like(gan_img)
        gan_img_l2[:, :, 54:] = gan_img[:, :, :-54]

        if mode == 0:
            random_number = random.random()
            img_0 = img
            if self.no_seg:
                target = mask_ones
            else:
                target = mask
            if random_number < 0.2:
                img_r2 = torch.zeros_like(img_in)
                img_r2[:, :, :-54] = img_in[:, :, 54:]
            else:
                img_r2 = torch.zeros_like(img)
                img_r2[:, :, :-54] = img[:, :, 54:]
            if random_number < 0.1:
                img_r1 = torch.zeros_like(img_in)
                img_r1[:, :, :-27] = img_in[:, :, 27:]
            else:
                img_r1 = torch.zeros_like(img)
                img_r1[:, :, :-27] = img[:, :, 27:]
            if random_number > 0.8:
                img_l2 = torch.zeros_like(img_in)
                img_l2[:, :, 54:] = img_in[:, :, :-54]
            else:
                img_l2 = torch.zeros_like(img)
                img_l2[:, :, 54:] = img[:, :, :-54]
            if random_number > 0.9:
                img_l1 = torch.zeros_like(img_in)
                img_l1[:, :, 27:] = img_in[:, :, :-27]
            else:
                img_l1 = torch.zeros_like(img)
                img_l1[:, :, 27:] = img[:, :, :-27]
        if mode == 1:
            possibilities = np.array([
                [1, 1, 1, 1],
                [1, 1, 1, 0],
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 0]
            ])
            random_number = random.randint(0, 9)
            use_in = possibilities[random_number, :]
            target = mask_zeros
            img_0 = img_in
            if use_in[3] == 1:
                img_r2 = torch.zeros_like(img_in)
                img_r2[:, :, :-54] = img_in[:, :, 54:]
            else:
                img_r2 = torch.zeros_like(img)
                img_r2[:, :, :-54] = img[:, :, 54:]
            if use_in[2] == 1:
                img_r1 = torch.zeros_like(img_in)
                img_r1[:, :, :-27] = img_in[:, :, 27:]
            else:
                img_r1 = torch.zeros_like(img)
                img_r1[:, :, :-27] = img[:, :, 27:]
            if use_in[0] == 1:
                img_l2 = torch.zeros_like(img_in)
                img_l2[:, :, 54:] = img_in[:, :, :-54]
            else:
                img_l2 = torch.zeros_like(img)
                img_l2[:, :, 54:] = img[:, :, :-54]
            if use_in[1] == 1:
                img_l1 = torch.zeros_like(img_in)
                img_l1[:, :, 27:] = img_in[:, :, :-27]
            else:
                img_l1 = torch.zeros_like(img)
                img_l1[:, :, 27:] = img[:, :, :-27]
        if mode == 2:
            possibilities = np.array([
                [1, 1, 1, 1],
                [1, 1, 1, 0],
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 0]
            ])
            random_number = random.randint(0, 9)
            use_2 = possibilities[random_number, :]
            target = mask_zeros
            img_0 = img_2
            if use_2[3] == 1:
                img_r2 = torch.zeros_like(img_2)
                img_r2[:, :, :-54] = img_2[:, :, 54:]
            else:
                img_r2 = torch.zeros_like(img)
                img_r2[:, :, :-54] = img[:, :, 54:]
            if use_2[2] == 1:
                img_r1 = torch.zeros_like(img_2)
                img_r1[:, :, :-27] = img_2[:, :, 27:]
            else:
                img_r1 = torch.zeros_like(img)
                img_r1[:, :, :-27] = img[:, :, 27:]
            if use_2[0] == 1:
                img_l2 = torch.zeros_like(img_2)
                img_l2[:, :, 54:] = img_2[:, :, :-54]
            else:
                img_l2 = torch.zeros_like(img)
                img_l2[:, :, 54:] = img[:, :, :-54]
            if use_2[1] == 1:
                img_l1 = torch.zeros_like(img_2)
                img_l1[:, :, 27:] = img_2[:, :, :-27]
            else:
                img_l1 = torch.zeros_like(img)
                img_l1[:, :, 27:] = img[:, :, :-27]


        if self.use_gan and self.use_neighbors:
            img_combined = torch.cat((img_l2, img_l1, img_0, img_r1, img_r2, gan_img_l2, gan_img_l1, gan_img, gan_img_r1, gan_img_r2), dim=0)
        elif self.use_gan and not self.use_neighbors:
            img_combined = torch.cat((img_0, gan_img), dim=0)
        elif not self.use_gan and self.use_neighbors:
            img_combined = torch.cat((img_l2, img_l1, img_0, img_r1, img_r2), dim=0)
        else:
            img_combined = img_0

        return img_combined, target, mask

    def __len__(self):
        return len(self.images)


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
            bytes_img = np.array(G_images.get(self.images[idx]))
            pil_image = Image.open(io.BytesIO(bytes_img))
            pil_image = pil_image.convert("RGB")
            img = np.array(pil_image)
            bytes_mask = np.array(G_masks.get(self.masks[idx]))
            pil_mask = Image.open(io.BytesIO(bytes_mask))
            pil_mask = pil_mask.convert("L")
            target = np.array(pil_mask)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

class RailSem19CroppedDatasetLikeFishyrails(torch.utils.data.Dataset):
    def __init__(self, data_path, mode="train", train_fraction=0.9):
        self.root = data_path
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
            bytes_img = np.array(G_images.get(self.images[idx]))
            pil_image = Image.open(io.BytesIO(bytes_img))
            pil_image = pil_image.convert("RGB")
            img = np.array(pil_image)
            bytes_mask = np.array(G_masks.get(self.masks[idx]))
            pil_mask = Image.open(io.BytesIO(bytes_mask))
            pil_mask = pil_mask.convert("L")
            target = np.array(pil_mask)


        img = torch.as_tensor(img)  # to torch
        img = img.permute((2, 0, 1))
        img = F.convert_image_dtype(img, torch.float)  # as float
        img = F.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # normalized

        target = torch.as_tensor(target, dtype=torch.int64)
        target = torch.unsqueeze(target, 0)

        return img, target, img.clone(), target.clone()

    def __len__(self):
        return len(self.images)

class FishyrailsCroppedDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.root = data_path
        with h5py.File(data_path, 'r') as hdf:
            G_images_fishy = hdf.get('images_fishy')
            self.images_fishy = sorted(G_images_fishy.keys())
            G_images_orig = hdf.get('images_orig')
            self.images_orig = sorted(G_images_orig.keys())
            G_masks_fishy = hdf.get('masks_fishy')
            self.masks_fishy = sorted(G_masks_fishy.keys())
            G_masks_orig = hdf.get('masks_orig')
            self.masks_orig = sorted(G_masks_orig.keys())

            # load all image files, sorting them to ensure that they are aligned
            #self.images = [image for image in sorted(os.listdir(os.path.join(self.root, "images"))) if image != ".gitignore"]
            #self.masks = [mask for mask in sorted(os.listdir(os.path.join(self.root, "masks"))) if mask != ".gitignore"]
            if len(self.masks_fishy) != len(self.images_fishy):
                print(f"ATTENTION: {len(self.images_fishy)} fishy images but {len(self.masks_fishy)} fishy masks!")

            self.image_fishy_data = dict()
            self.image_orig_data = dict()
            self.mask_fishy_data = dict()
            self.mask_orig_data = dict()
            for image_fishy_name, image_orig_name, mask_fishy_name, mask_orig_name in zip(self.images_fishy, self.images_orig, self.masks_fishy, self.masks_orig):
                bytes_img_fishy = np.array(G_images_fishy.get(image_fishy_name))
                pil_image_fishy = Image.open(io.BytesIO(bytes_img_fishy))
                pil_image_fishy = pil_image_fishy.convert("RGB")
                self.image_fishy_data[image_fishy_name] = np.array(pil_image_fishy)

                bytes_mask_fishy = np.array(G_masks_fishy.get(mask_fishy_name))
                pil_mask_fishy = Image.open(io.BytesIO(bytes_mask_fishy))
                pil_mask_fishy = pil_mask_fishy.convert("L")
                self.mask_fishy_data[mask_fishy_name] = np.array(pil_mask_fishy)

                bytes_img_orig = np.array(G_images_orig.get(image_orig_name))
                pil_image_orig = Image.open(io.BytesIO(bytes_img_orig))
                pil_image_orig = pil_image_orig.convert("RGB")
                self.image_orig_data[image_orig_name] = np.array(pil_image_orig)

                bytes_mask_orig = np.array(G_masks_orig.get(mask_orig_name))
                pil_mask_orig = Image.open(io.BytesIO(bytes_mask_orig))
                pil_mask_orig = pil_mask_orig.convert("L")
                self.mask_orig_data[mask_orig_name] = np.array(pil_mask_orig)
                # self.image_fishy_data[image_fishy_name] = np.array(G_images_fishy.get(image_fishy_name))
                # self.image_orig_data[image_orig_name] = np.array(G_images_orig.get(image_orig_name))
                # self.mask_fishy_data[mask_fishy_name] = np.array(G_masks_fishy.get(mask_fishy_name))
                # self.mask_orig_data[mask_orig_name] = np.array(G_masks_orig.get(mask_orig_name))
        print(f"FishyrailsCropped Dataset: {len(self.images_fishy)} images from {self.root}")

    def __getitem__(self, idx):
        # load images and masks
        # img_path = os.path.join(self.root, "images", self.images[idx])
        # mask_path = os.path.join(self.root, "masks", self.masks[idx])
        # img = Image.open(img_path).convert("RGB")
        # target = Image.open(mask_path) # no need to convert masks to RGB

        # Transform data to torch and normalize

        img_fishy = self.image_fishy_data[self.images_fishy[idx]]
        img_fishy = torch.as_tensor(img_fishy) # to torch
        img_fishy = img_fishy.permute((2, 0, 1))
        img_fishy = F.convert_image_dtype(img_fishy, torch.float) # as float
        img_fishy = F.normalize(img_fishy, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # normalized

        img_orig = self.image_orig_data[self.images_orig[idx]]
        img_orig = torch.as_tensor(img_orig) # to torch
        img_orig = img_orig.permute((2, 0, 1))
        img_orig = F.convert_image_dtype(img_orig, torch.float) # as float
        img_orig = F.normalize(img_orig, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # normalized

        target_fishy = self.mask_fishy_data[self.masks_fishy[idx]]
        target_fishy = torch.as_tensor(target_fishy, dtype=torch.int64)
        target_fishy = torch.unsqueeze(target_fishy, 0)

        target_orig = self.mask_orig_data[self.masks_orig[idx]]
        target_orig = torch.as_tensor(target_orig, dtype=torch.int64)
        target_orig = torch.unsqueeze(target_orig, 0)

        return img_fishy, target_fishy, img_orig, target_orig

    def __len__(self):
        return len(self.images_fishy)



class ImageNetDatasetSegmentation(torch.utils.data.Dataset):
    def __init__(self, data_path, transforms, mode="train", train_fraction=0.9):
        self.root = data_path
        self.transforms = transforms
        self.data_path = data_path
        with h5py.File(data_path, 'r') as hdf:
            G_images = hdf.get('images')
            self.images = list(G_images.keys())
            random.shuffle(self.images)

            # Train validation split
            train_length = int(len(self.images) * train_fraction)
            if mode == "train":
                self.images = self.images[:train_length]
            elif mode == "val":
                self.images = self.images[train_length:]
            else:
                mode = "full"  # take all images

            self.image_data = dict()
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
            bytes_img = np.array(G_images.get(self.images[idx]))
            pil_image = Image.open(io.BytesIO(bytes_img))
            # make sure image is RGB (necessary for grayscale or 4 channel images)
            pil_image = pil_image.convert("RGB")
            img = np.array(pil_image)
            H, W, C = img.shape
            assert C == 3, f"Channels != 3: Image {self.images[idx]} shape: {img.shape}"
            target = np.zeros((H,W))

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

class ImageNetDatasetPatch(torch.utils.data.Dataset):
    def __init__(self, data_path, mode="train", train_fraction=0.9, patch_size=32, grayscale_ratio=0.1):
        self.root = data_path
        self.data_path = data_path
        self.patch_size = patch_size
        self.grayscale_ratio = grayscale_ratio

        # Image Net dataset
        with h5py.File(data_path, 'r') as hdf:
            G_images = hdf.get('images')
            self.images = list(G_images.keys())
            random.shuffle(self.images)

        # Train validation split
        train_length = int(len(self.images) * train_fraction)
        if mode == "train":
            self.images = self.images[:train_length]
        elif mode == "val":
            self.images = self.images[train_length:]
        else:
            mode = "full"  # take all images

        print(f"ImageNet Patch Dataset {mode}: {len(self.images)} images from {self.root}")

    def __getitem__(self, idx):
        img_idx = random.randint(0, len(self.images) - 1)
        img_idx_2 = img_idx
        while img_idx_2 == img_idx:
            img_idx_2 = random.randint(0, len(self.images) - 1)

        with h5py.File(self.data_path, 'r') as hdf:
            # image
            G_images = hdf.get('images')
            bytes_img = np.array(G_images.get(self.images[img_idx]))
            pil_image = Image.open(io.BytesIO(bytes_img))
            pil_image = pil_image.convert("RGB")
            img = np.array(pil_image)

            bytes_img_2 = np.array(G_images.get(self.images[img_idx_2]))
            pil_image_2 = Image.open(io.BytesIO(bytes_img_2))
            pil_image_2 = pil_image_2.convert("RGB")
            img_2 = np.array(pil_image_2)


        # Grayscale ?
        if random.random() < self.grayscale_ratio:
            grayscale = True
        else:
            grayscale = False

        # Crop parameters
        # image sizes
        image_size_orig = random.randint(4*self.patch_size, 16*self.patch_size)
        image_size_minus = random.randint(4 * self.patch_size, 16 * self.patch_size)

        # patch locations
        patch_orig_left = int(self.patch_size*3/4) + random.randint(0, image_size_orig - 1 - int(self.patch_size*6/4))
        patch_orig_top = int(self.patch_size*3/4) + random.randint(0, image_size_orig - 1 - int(self.patch_size*6/4))
        if random.random() < 0.5:
            patch_plus_left = patch_orig_left + random.randint(0, int(self.patch_size/4))
            patch_plus_top = patch_orig_top + random.randint(0, int(self.patch_size/4))
        else:
            patch_plus_left = patch_orig_left - random.randint(0, int(self.patch_size / 4))
            patch_plus_top = patch_orig_top - random.randint(0, int(self.patch_size / 4))
        patch_minus_left = int(self.patch_size/2) + random.randint(0, image_size_minus - 1 - self.patch_size)
        patch_minus_top = int(self.patch_size / 2) + random.randint(0, image_size_minus - 1 - self.patch_size)

        # Apply Crop and Augmentation Transforms
        patch_orig = transforms.imagenet_crop_augment(img, image_size_orig, self.patch_size, patch_orig_left, patch_orig_top, augment=False, grayscale=grayscale)
        patch_plus = transforms.imagenet_crop_augment(img, image_size_orig, self.patch_size, patch_plus_left, patch_plus_top, augment=True, grayscale=grayscale)
        patch_minus = transforms.imagenet_crop_augment(img_2, image_size_minus, self.patch_size, patch_minus_left, patch_minus_top, augment=False, grayscale=grayscale)

        return patch_orig, patch_plus, patch_minus

    def __len__(self):
        return len(self.images)


class RealWorldDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.root = data_path
        all_files = [f for f in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, f))]
        masks = [os.path.join(self.root, f) for f in all_files if f.endswith("mask.png")]
        images = [os.path.join(self.root, f) for f in all_files if
                  (f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")) and (not f.endswith("mask.png") and not f.endswith("obstacle.png"))]
        names = [os.path.splitext(f)[0] for f in all_files if
                 (f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")) and (not f.endswith("mask.png") and not f.endswith("obstacle.png"))]
        annotations = [os.path.join(self.root, f) for f in all_files if f.endswith(".txt")]
        if len(images) != len(masks) or len(annotations) != len(images):
            print(f"Error: {len(images)} Images but {len(masks)} masks and {len(annotations)} annotations!")
            return
        self.masks = sorted(masks)
        self.images = sorted(images)
        self.names = sorted(names)
        print(self.names)
        self.annotations = sorted(annotations)

        print(f"Real World Dataset: {len(self.images)} images from {self.root}")

    def __getitem__(self, idx):
        pil_img = Image.open(self.images[idx])
        pil_img = pil_img.convert("RGB")
        img = np.array(pil_img)
        pil_mask = Image.open(self.masks[idx])
        pil_mask = pil_mask.convert("L")
        mask = np.array(pil_mask)
        with open(self.annotations[idx]) as f:
            annotation = np.loadtxt(f)

        img = torch.as_tensor(img) # to torch
        img = img.permute((2, 0, 1))
        img = F.convert_image_dtype(img, torch.float) # as float
        img = F.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # normalized

        mask = torch.as_tensor(mask, dtype=torch.int64)
        mask = torch.unsqueeze(mask, 0)

        name = self.names[idx]

        return img, mask, annotation, name

    def __len__(self):
        return len(self.images)

class RealWorldDataset2(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.root = data_path
        all_files = [f for f in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, f))]
        masks = [os.path.join(self.root, f) for f in all_files if f.endswith("mask.png")]
        annotations = [os.path.join(self.root, f) for f in all_files if f.endswith("obstacle.png")]
        images = [os.path.join(self.root, f) for f in all_files if
                  (f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")) and (not f.endswith("mask.png") and not f.endswith("obstacle.png"))]
        names = [os.path.splitext(f)[0] for f in all_files if
                 (f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")) and (not f.endswith("mask.png") and not f.endswith("obstacle.png"))]
        if len(images) != len(masks) or len(annotations) != len(images):
            print(f"Error: {len(images)} Images but {len(masks)} masks and {len(annotations)} annotations!")
            return
        self.masks = sorted(masks)
        self.images = sorted(images)
        self.names = sorted(names)
        print(self.names)
        self.annotations = sorted(annotations)

        print(f"Real World Dataset: {len(self.images)} images from {self.root}")

    def __getitem__(self, idx):
        pil_img = Image.open(self.images[idx])
        pil_img = pil_img.convert("RGB")
        img = np.array(pil_img)
        pil_mask = Image.open(self.masks[idx])
        pil_mask = pil_mask.convert("L")
        mask = np.array(pil_mask)
        pil_anno = Image.open(self.annotations[idx])
        pil_anno = pil_anno.convert("L")
        anno = np.array(pil_anno)

        img = torch.as_tensor(img) # to torch
        img = img.permute((2, 0, 1))
        img = F.convert_image_dtype(img, torch.float) # as float
        img = F.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # normalized

        mask = torch.as_tensor(mask, dtype=torch.int64)
        mask = torch.unsqueeze(mask, 0)

        anno = torch.as_tensor(anno, dtype=torch.int64)
        anno = torch.unsqueeze(anno, 0)

        name = self.names[idx]

        return img, mask, anno, name

    def __len__(self):
        return len(self.images)

class RealWorldDatasetRaw(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.root = data_path
        all_files = [f for f in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, f))]
        # masks = [os.path.join(self.root, f) for f in all_files if f.endswith("mask.png")]
        images = [os.path.join(self.root, f) for f in all_files if
                  (f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")) and not f.endswith("mask.png")]
        names = [os.path.splitext(f)[0] for f in all_files if
                 (f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")) and not f.endswith("mask.png")]
        self.images = sorted(images)
        self.names = sorted(names)
        print(self.names)

        print(f"Real World Dataset: {len(self.images)} images from {self.root}")

    def __getitem__(self, idx):
        pil_img = Image.open(self.images[idx])
        pil_img = pil_img.convert("RGB")
        size = pil_img.size
        pil_img = pil_img.resize((224, 224))
        img = np.array(pil_img)

        img = torch.as_tensor(img) # to torch
        img = img.permute((2, 0, 1))
        img = F.convert_image_dtype(img, torch.float) # as float
        img = F.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # normalized

        name = self.names[idx]

        return img, name, size

    def __len__(self):
        return len(self.images)
