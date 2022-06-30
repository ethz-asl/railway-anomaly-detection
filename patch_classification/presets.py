import torch
import transforms as T
import numpy as np
from PIL import Image


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [
            T.NpToTensor(),
            T.RandomResize(min_size, max_size)
        ]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend(
            [
                T.RandomCrop(crop_size),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)

class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), crop=False):
        if crop:
            self.transforms = T.Compose(
                [
                    T.NpToTensor(),
                    T.RandomResize(base_size, base_size),
                    T.RandomCrop(base_size),
                    T.ConvertImageDtype(torch.float),
                    T.Normalize(mean=mean, std=std),
                ])
        else:
            self.transforms = T.Compose(
                [
                    T.NpToTensor(),
                    T.RandomResize(base_size, base_size),
                    T.ConvertImageDtype(torch.float),
                    T.Normalize(mean=mean, std=std),
                ])

    def __call__(self, img, target):
        return self.transforms(img, target)

class GanSegmentationPreset:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), flip=0, crop=0):
        trans = [
                    T.NpToTensor(),
                    T.RandomResize(base_size, base_size),
                ]
        if flip > 0:
            trans.append(T.RandomHorizontalFlip(flip_prob=flip))
        if crop > 0:
            trans.append(T.RandomCropAndRescale(crop_prob=crop, min_size=int(base_size/2), max_size=base_size, base_size=base_size))
        trans.extend(
            [
                    T.ConvertImageDtype(torch.float),
                    T.Normalize(mean=mean, std=std)
                ]
        )
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class AutoencoderPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend(
            [
                T.RandomCrop(crop_size),
                T.PILToTensorAA(),
                T.ConvertImageDtypeAA(torch.float),
                T.Normalize(mean=mean, std=std), # does not normalize target!
            ]
        )
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class AutoencoderPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose(
            [
                T.RandomResizeAA(base_size, base_size),
                T.PILToTensorAA(),
                T.ConvertImageDtypeAA(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img, target):
        return self.transforms(img, target)


def denormalize(image, target, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    trans = [T.DeNormalize(mean, std)]
    transforms = T.Compose(trans)
    return transforms(image, target)

def denormalize_tanh(image, target, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    trans = [T.DeNormalize(mean, std)]
    transforms = T.Compose(trans)
    image, target = transforms(image, target)
    image = (image - 0.5) * 2
    return image, target

def normalize(image, target, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    trans = [T.Normalize(mean, std)]
    transforms = T.Compose(trans)
    return transforms(image, target)

def re_transform(image, target, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # target is kept unchanged
    trans = [T.DeNormalize(mean, std),
             T.ConvertImageDtype(torch.uint8),
             T.TensorToPIL()]
    transforms = T.Compose(trans)
    return transforms(image, target)

def re_convert(image, target, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # target is kept unchanged
    trans = [T.ConvertImageDtype(torch.uint8),
             T.TensorToPIL()]
    transforms = T.Compose(trans)
    return transforms(image, target)

def re_convert_tanh(image, target, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # target is kept unchanged
    image = (image / 2) + 0.5
    trans = [T.ConvertImageDtype(torch.uint8),
             T.TensorToPIL()]
    transforms = T.Compose(trans)
    return transforms(image, target)

def torch_mask_to_pil(mask):
    mask = torch.squeeze(mask)
    mask = mask.cpu().numpy()
    mask = np.uint8(mask)
    mask = Image.fromarray(mask, mode='L')
    return mask
