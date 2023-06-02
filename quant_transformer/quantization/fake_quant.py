import torch
import torch.nn as nn
from .observer import MinMaxObserver
from .util_quant import (
    fake_quantize_per_channel_affine,
    fake_quantize_per_tensor_affine,
    fake_quantize_learnable_per_tensor_affine_training,
    fake_quantize_learnable_per_channel_affine_training,
    fake_quantize_learnableplus_per_channel_affine_training,
    fake_quantize_learnableplus_per_tensor_affine_training,
)


class QuantizeBase(nn.Module):

    def __init__(self, observer=MinMaxObserver, bit=8, symmetric=False, ch_axis=-1):
        super().__init__()
        self.observer = observer(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.bit = bit
        self.symmetric = symmetric
        self.ch_axis = ch_axis
        self.observer_enabled = 0
        self.fake_quant_enabled = 0
        self.quant_min = self.observer.quant_min
        self.quant_max = self.observer.quant_max

    def set_name(self, name):
        self.name = name

    @torch.jit.export
    def calculate_qparams(self):
        return self.observer.calculate_qparams()

    @torch.jit.export
    def disable_observer(self):
        self.observer_enabled = 0

    @torch.jit.export
    def enable_observer(self):
        self.observer_enabled = 1

    @torch.jit.export
    def disable_fake_quant(self):
        self.fake_quant_enabled = 0

    @torch.jit.export
    def enable_fake_quant(self):
        self.fake_quant_enabled = 1

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'symmetric={}, bit={}, ch_axis={}, quant_min={}, quant_max={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.symmetric, self.bit, self.ch_axis,
                   self.quant_min, self.quant_max)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Removing this function throws an error that the the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                if name == 'scale':
                    if isinstance(self.scale, nn.Parameter):
                        self.scale.data = torch.ones_like(val.to(self.scale.device))
                    else:
                        self.scale.resize_(val.shape)
                else:
                    assert name == 'zero_point'
                    if isinstance(self.zero_point, nn.Parameter):
                        self.zero_point.data = torch.ones_like(val.to(self.zero_point.device))
                    else:
                        self.zero_point.resize_(val.shape)
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == 'scale':
                        self.scale.copy_(val)
                    else:
                        assert name == 'zero_point'
                        self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class FixedFakeQuantize(QuantizeBase):

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))

    def forward(self, X, observation_mask=None, seq_pos=-1):
        if self.observer_enabled == 1:
            self.observer(X.detach(), observation_mask=observation_mask, seq_pos=seq_pos)
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled == 1:
            if self.ch_axis != -1:
                X = fake_quantize_per_channel_affine(
                    X, self.scale.data, self.zero_point.data.int(), self.ch_axis,
                    self.quant_min, self.quant_max)
            else:
                X = fake_quantize_per_tensor_affine(
                    X, self.scale.item(), self.zero_point.item(),
                    self.quant_min, self.quant_max)
        return X


class LSQFakeQuantize(QuantizeBase):
    # used for symmetric quantization scheme
    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1, use_grad_scaling=True):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float).eps]))
        self.use_grad_scaling = use_grad_scaling

    def forward(self, X, observation_mask=None, seq_pos=-1):
        if self.observer_enabled == 1:
            self.observer(X.detach(), observation_mask=observation_mask, seq_pos=seq_pos)
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.ch_axis != -1:
                self.scale.data = torch.ones_like(_scale)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.data.copy_(_scale)
            self.zero_point.copy_(_zero_point)
        else:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())
        if self.fake_quant_enabled == 1:
            if self.ch_axis != -1:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() / X.shape[self.ch_axis] * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnable_per_channel_affine_training(
                    X, self.scale, self.zero_point.data.int(), self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
            else:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnable_per_tensor_affine_training(
                    X, self.scale, self.zero_point.item(), self.quant_min, self.quant_max, grad_factor)
        return X


class LSQPlusFakeQuantize(QuantizeBase):
    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1, use_grad_scaling=True):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float))
        self.zero_point = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float))
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float).eps]))
        self.use_grad_scaling = use_grad_scaling

    def forward(self, X, observation_mask=None, seq_pos=-1):
        if self.observer_enabled == 1:
            self.observer(X.detach(), observation_mask=observation_mask, seq_pos=seq_pos)
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.ch_axis != -1:
                self.scale.data = torch.ones_like(_scale)
                self.zero_point.data = torch.zeros_like(_zero_point.float())
            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point.float())
        else:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())
            self.zero_point.data.clamp_(self.quant_min, self.quant_max).float()

        if self.fake_quant_enabled == 1:
            if self.ch_axis != -1:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() / X.shape[self.ch_axis] * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnableplus_per_channel_affine_training(
                    X, self.scale, self.zero_point, self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
            else:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnableplus_per_tensor_affine_training(
                    X, self.scale, self.zero_point, self.quant_min, self.quant_max, grad_factor)
        return X
