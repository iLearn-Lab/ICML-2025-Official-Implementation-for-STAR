import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from star.algos.utils.obs_core import CropRandomizer


class TranslationAug(nn.Module):
    """Apply translation augmentation through padded random crops."""

    def __init__(
        self,
        shape_meta,
        translation,
    ):
        super().__init__()

        self.randomizers = nn.ModuleDict()
        self.shape_meta = shape_meta
        self.pad_translation = translation // 2

        for name, input_shape in shape_meta['observation']['rgb'].items():
            input_shape = tuple(input_shape)
            pad_output_shape = (
                input_shape[0],
                input_shape[1] + translation,
                input_shape[2] + translation,
            )

            crop_randomizer = CropRandomizer(
                input_shape=pad_output_shape,
                crop_height=input_shape[1],
                crop_width=input_shape[2],
            )
            self.randomizers[name] = crop_randomizer

    def forward(self, data):
        if self.training:
            for name in self.shape_meta['observation']['rgb']:
                x = data['obs'][name]
                batch_size, temporal_len, img_c, img_h, img_w = x.shape
                crop_randomizer = self.randomizers[name]

                x = x.reshape(batch_size, temporal_len * img_c, img_h, img_w)
                out = F.pad(x, pad=(self.pad_translation,) * 4, mode="replicate")
                out = crop_randomizer.forward_in(out)
                out = out.reshape(batch_size, temporal_len, img_c, img_h, img_w)
                data['obs'][name] = out
        return data


class BatchWiseImgColorJitterAug(torch.nn.Module):
    """Apply color jitter to a random subset of batch elements."""

    def __init__(
        self,
        shape_meta,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.1,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.epsilon = epsilon
        self.shape_meta = shape_meta

    def forward(self, data):
        if self.training:
            for name in self.shape_meta['observation']['rgb']:
                x = data['obs'][name]
                mask = torch.rand((x.shape[0], *(1,) * (len(x.shape) - 1)), device=x.device) > self.epsilon
                jittered = self.color_jitter(x)
                out = mask * jittered + torch.logical_not(mask) * x
                data['obs'][name] = out
        return data


class DataAugGroup(nn.Module):
    """Compose multiple observation augmentations."""

    def __init__(self, aug_list, shape_meta):
        super().__init__()
        aug_list = [aug(shape_meta) for aug in aug_list]
        self.aug_layer = nn.Sequential(*aug_list)

    def forward(self, data):
        return self.aug_layer(data)
