import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

from torch import Tensor
from typing import List, Tuple, Any, Optional


def pad_if_smaller(img, size, fill=0):
    min_size = min([img.shape[1], img.shape[2]])
    if min_size < size:
        ow, oh = img.shape[2], img.shape[1]
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target

class ResizeCrop:
    def __init__(self, image_size, patch_size, left, up):
        self.image_size = image_size
        self.patch_size = patch_size
        self.left = left
        self.up = up

    def __call__(self, image, target):
        image = F.resize(image, self.image_size)
        image = F.crop(image, top=self.up, left=self.left, height=self.patch_size, width=self.patch_size)
        return image, target

class RandomResizeAA:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

class RandomCropAndRescale:
    def __init__(self, crop_prob, min_size, max_size, base_size):
        self.min_size = min_size
        self.max_size = max_size
        self.base_size = base_size
        self.crop_prob = crop_prob

    def __call__(self, image, target):
        if random.random() < self.crop_prob:
            size = random.randint(self.min_size, self.max_size)
            crop_params = T.RandomCrop.get_params(image, (size, size))
            image = F.crop(image, *crop_params)
            target = F.crop(target, *crop_params)
            image = F.resize(image, self.base_size)
            target = F.resize(target, self.base_size)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target

class NpToTensor:
    def __call__(self, image, target):
        image = torch.as_tensor(image)
        image = image.permute((2, 0, 1))
        target = torch.as_tensor(target, dtype=torch.int64)
        target = torch.unsqueeze(target, 0)
        return image, target

class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

class PILToTensorAA:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = F.pil_to_tensor(target)
        return image, target

class TensorToPIL:
    def __init__(self, mode=None):
        self.mode = mode
    def __call__(self, image, target):
        image = torch.squeeze(image)
        image = F.to_pil_image(image, mode=self.mode)
        return image, target


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target

class ConvertImageDtypeAA:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        target = F.convert_image_dtype(target, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class DeNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = denormalize(image, mean=self.mean, std=self.std)
        return image, target


def denormalize(tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False) -> Tensor:
    """Normalize a float tensor image with mean and standard deviation.
    This transform does not support PIL Image.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Float tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if not tensor.is_floating_point():
        raise TypeError('Input tensor should be a float tensor. Got {}.'.format(tensor.dtype))

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(
            'std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    # tensor.sub_(mean).div_(std)
    tensor.mul_(std).add_(mean)
    return tensor


def imagenet_crop_augment(image, image_size, patch_size, patch_left, patch_top, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), augment=False, grayscale=False):
    # Np to tensor
    image = torch.as_tensor(image)
    image = image.permute((2, 0, 1))
    # resize
    image = F.resize(image, image_size)
    image = F.crop(image, top=patch_top, left=patch_left, height=patch_size, width=patch_size)
    # convert to float
    image = F.convert_image_dtype(image, torch.float)
    if augment:
        # change brightness
        brightness_change = torch.randn((1)) * 0.1
        image[0, ::] += brightness_change
        image[1, ::] += brightness_change
        image[2, ::] += brightness_change
        # add noise
        image = image + torch.randn(image.size()) * 0.1
        image[image > 1] = 1
        image[image < 0] = 0
    if grayscale:
        # convert to grayscale
        gray_image = 0.299 * image[0, ::] + 0.587 * image[1, ::] + 0.114 * image[2, ::]
        image[0, ::] = gray_image
        image[1, ::] = gray_image
        image[2, ::] = gray_image

    # normalize
    image = F.normalize(image, mean=mean, std=std)

    return image

