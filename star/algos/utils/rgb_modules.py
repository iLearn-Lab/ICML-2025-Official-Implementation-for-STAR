"""Vision encoders used by STAR."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class SpatialSoftmax(nn.Module):
    """Spatial softmax keypoint extractor."""

    def __init__(self, in_channels, in_height, in_width, num_keypoints=None):
        super().__init__()
        self._in_channels = in_channels
        self._in_height = in_height
        self._in_width = in_width
        self._num_keypoints = in_channels if num_keypoints is None else num_keypoints
        self._spatial_conv = nn.Conv2d(in_channels, self._num_keypoints, kernel_size=1)

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, in_width).float(),
            torch.linspace(-1, 1, in_height).float(),
            indexing="ij",
        )
        self.register_buffer("pos_x", pos_x.reshape(1, in_width * in_height))
        self.register_buffer("pos_y", pos_y.reshape(1, in_width * in_height))

    def forward(self, x):
        if x.shape[1:] != (self._in_channels, self._in_height, self._in_width):
            raise ValueError(
                "SpatialSoftmax received an input with shape "
                f"{tuple(x.shape[1:])}, expected "
                f"{(self._in_channels, self._in_height, self._in_width)}."
            )

        if self._num_keypoints != self._in_channels:
            x = self._spatial_conv(x)

        attention = F.softmax(x.reshape(-1, self._in_height * self._in_width), dim=-1)
        keypoint_x = (self.pos_x * attention).sum(1, keepdims=True).view(-1, self._num_keypoints)
        keypoint_y = (self.pos_y * attention).sum(1, keepdims=True).view(-1, self._num_keypoints)
        return torch.cat([keypoint_x, keypoint_y], dim=1)


class SpatialProjection(nn.Module):
    """Project convolutional feature maps to a fixed-size embedding."""

    def __init__(self, input_shape, out_dim):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError("SpatialProjection expects input_shape=(C, H, W).")

        in_channels, in_height, in_width = input_shape
        num_keypoints = out_dim // 2
        self.out_dim = out_dim
        self.spatial_softmax = SpatialSoftmax(in_channels, in_height, in_width, num_keypoints=num_keypoints)
        self.projection = nn.Linear(num_keypoints * 2, out_dim)

    def forward(self, x):
        return self.projection(self.spatial_softmax(x))

    def output_shape(self, input_shape):
        return input_shape[:-3] + (self.out_dim,)


class ResnetEncoder(nn.Module):
    """ResNet-18 image encoder with optional FiLM conditioning."""

    def __init__(
        self,
        input_shape,
        output_size,
        pretrained=False,
        freeze=False,
        remove_layer_num=2,
        no_stride=False,
        language_dim=768,
        language_fusion="film",
        do_projection=True,
    ):
        super().__init__()

        if remove_layer_num > 5:
            raise ValueError("remove_layer_num must be <= 5.")
        if len(input_shape) != 3:
            raise ValueError("ResnetEncoder expects input_shape=(C, H, W).")

        in_channels = input_shape[0]
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        layers = list(torchvision.models.resnet18(weights=weights).children())[:-remove_layer_num]

        if in_channels != 3:
            if freeze:
                raise ValueError("Cannot freeze a pretrained ResNet when the input channel count differs from 3.")
            layers[0] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        if no_stride:
            layers[0].stride = (1, 1)
            layers[3].stride = 1

        self.resnet18_base = nn.Sequential(*layers[:4])
        self.block_1 = layers[4][0]
        self.block_2 = layers[4][1]
        self.block_3 = layers[5][0]
        self.block_4 = layers[5][1]

        self.language_fusion = language_fusion
        if language_fusion != "none":
            self.lang_proj1 = nn.Linear(language_dim, 64 * 2)
            self.lang_proj2 = nn.Linear(language_dim, 64 * 2)
            self.lang_proj3 = nn.Linear(language_dim, 128 * 2)
            self.lang_proj4 = nn.Linear(language_dim, 128 * 2)

        if freeze:
            for module in (self.resnet18_base, self.block_1, self.block_2, self.block_3, self.block_4):
                for param in module.parameters():
                    param.requires_grad = False

        self.normalizer = (
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if pretrained
            else nn.Identity()
        )

        with torch.no_grad():
            sample = torch.zeros(1, *input_shape)
            sample = self.block_4(self.block_3(self.block_2(self.block_1(self.resnet18_base(sample)))))

        if do_projection:
            self.projection_layer = SpatialProjection(tuple(sample.shape[1:]), output_size)
            self.out_channels = output_size
        else:
            self.projection_layer = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
            self.out_channels = sample.shape[1]

    def _apply_film(self, x, projection, language_embedding):
        batch_size, channels, _, _ = x.shape
        beta, gamma = torch.split(
            projection(language_embedding).reshape(batch_size, channels * 2, 1, 1),
            [channels, channels],
            dim=1,
        )
        return (1 + gamma) * x + beta

    def forward(self, x, langs=None):
        x = self.normalizer(x)
        x = self.resnet18_base(x)

        x = self.block_1(x)
        if langs is not None and self.language_fusion != "none":
            x = self._apply_film(x, self.lang_proj1, langs)

        x = self.block_2(x)
        if langs is not None and self.language_fusion != "none":
            x = self._apply_film(x, self.lang_proj2, langs)

        x = self.block_3(x)
        if langs is not None and self.language_fusion != "none":
            x = self._apply_film(x, self.lang_proj3, langs)

        x = self.block_4(x)
        if langs is not None and self.language_fusion != "none":
            x = self._apply_film(x, self.lang_proj4, langs)

        return self.projection_layer(x)
