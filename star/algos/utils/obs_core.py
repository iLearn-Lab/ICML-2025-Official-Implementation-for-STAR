"""Observation randomizers used by image augmentation."""

import abc

import torch
import torch.nn as nn

import star.utils.obs_utils as ObsUtils
import star.utils.tensor_utils as TensorUtils


class Randomizer(nn.Module):
    """Base class for randomizers applied to raw observations."""

    def __init__(self):
        super().__init__()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ObsUtils.register_randomizer(cls)

    def output_shape(self, input_shape=None):
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape_in(self, input_shape=None):
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape_out(self, input_shape=None):
        raise NotImplementedError

    def forward_in(self, inputs):
        if self.training:
            return self._forward_in(inputs)
        return self._forward_in_eval(inputs)

    def forward_out(self, inputs):
        if self.training:
            return self._forward_out(inputs)
        return self._forward_out_eval(inputs)

    @abc.abstractmethod
    def _forward_in(self, inputs):
        raise NotImplementedError

    def _forward_in_eval(self, inputs):
        return inputs

    @abc.abstractmethod
    def _forward_out(self, inputs):
        raise NotImplementedError

    def _forward_out_eval(self, inputs):
        return inputs

    @abc.abstractmethod
    def _visualize(self, pre_random_input, randomized_input, num_samples_to_visualize=2):
        raise NotImplementedError


class CropRandomizer(Randomizer):
    """Randomly crop images during training and center-crop them at evaluation."""

    def __init__(
        self,
        input_shape,
        crop_height=76,
        crop_width=76,
        num_crops=1,
        pos_enc=False,
    ):
        super().__init__()

        assert len(input_shape) == 3
        assert crop_height < input_shape[1]
        assert crop_width < input_shape[2]

        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def output_shape_in(self, input_shape=None):
        out_channels = self.input_shape[0] + 2 if self.pos_enc else self.input_shape[0]
        return [out_channels, self.crop_height, self.crop_width]

    def output_shape_out(self, input_shape=None):
        return list(input_shape)

    def _forward_in(self, inputs):
        assert len(inputs.shape) >= 3
        cropped, _ = ObsUtils.sample_random_image_crops(
            images=inputs,
            crop_height=self.crop_height,
            crop_width=self.crop_width,
            num_crops=self.num_crops,
            pos_enc=self.pos_enc,
        )
        return TensorUtils.join_dimensions(cropped, 0, 1)

    def _forward_in_eval(self, inputs):
        assert len(inputs.shape) >= 3
        inputs = inputs.permute(*range(inputs.dim() - 3), inputs.dim() - 2, inputs.dim() - 1, inputs.dim() - 3)
        cropped = ObsUtils.center_crop(inputs, self.crop_height, self.crop_width)
        return cropped.permute(*range(cropped.dim() - 3), cropped.dim() - 1, cropped.dim() - 3, cropped.dim() - 2)

    def _forward_out(self, inputs):
        batch_size = inputs.shape[0] // self.num_crops
        out = TensorUtils.reshape_dimensions(
            inputs,
            begin_axis=0,
            end_axis=0,
            target_dims=(batch_size, self.num_crops),
        )
        return out.mean(dim=1)

    def _visualize(self, pre_random_input, randomized_input, num_samples_to_visualize=2):
        batch_size = pre_random_input.shape[0]
        random_sample_inds = torch.randint(0, batch_size, size=(num_samples_to_visualize,))
        _ = TensorUtils.to_numpy(pre_random_input)[random_sample_inds]
        randomized_input = TensorUtils.reshape_dimensions(
            randomized_input,
            begin_axis=0,
            end_axis=0,
            target_dims=(batch_size, self.num_crops),
        )
        _ = TensorUtils.to_numpy(randomized_input[random_sample_inds])

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"input_shape={self.input_shape}, "
            f"crop_size=[{self.crop_height}, {self.crop_width}], "
            f"num_crops={self.num_crops})"
        )
