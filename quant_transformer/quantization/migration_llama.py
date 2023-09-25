import math
import torch
import torch.nn as nn
import logging
from scipy.optimize import minimize_scalar
from .fake_quant import QuantizeBase
from .observer import MinMaxObserver
from .util_quant import fake_quantize_per_channel_affine, fake_quantize_per_tensor_affine
import quant_transformer.model.quant_llama as quant_llama
# QuantizedLlamaRMSNorm, apply_rotary_pos_emb, rotate_half
logger = logging.getLogger('OS+')
scale_list = []


def migration(act, weight, a_qconfig, w_qconfig, module_type, extra_dict=None):
    migrator = Migrator1DRangeSearch(act, weight, a_qconfig, w_qconfig, module_type, extra_dict)
    best_scale = migrator()
    scale_list.append(best_scale)
    return best_scale


def fuse_migration(model):
    cnt = 0
    disable_down_proj = True
    for name, module in model.named_modules():
        if 'down_proj.act_fake_quant' in name:
            disable_down_proj = False
    for name, module in model.named_modules():
        if isinstance(module, quant_llama.QuantizedLlamaRMSNorm):
            if cnt < len(scale_list):
                module.weight.data /= scale_list[cnt]
                cnt += 1
        if not disable_down_proj:
            if 'up_proj.module.weight_fake_quant' in name and isinstance(module, QuantizeBase):
                if cnt < len(scale_list):
                    module.scale.data = (module.scale.data / scale_list[cnt]).reshape(-1, 1)
                    cnt += 1


class MigratorBase(nn.Module):

    def __init__(self, input, weight, a_qconfig, w_qconfig, module_type, extra_dict=None):
        super().__init__()
        self.input = input
        self.weight = weight
        self.a_qconfig = a_qconfig
        self.w_qconfig = w_qconfig
        self.module_type = module_type
        self.extra_dict = extra_dict
        self.dtype = self.input.dtype
        self.device = self.input.device
        # calculate min max in advance
        self.cmx = self.input.max(0)[0].max(0)[0]
        self.cmn = self.input.min(0)[0].min(0)[0]
        self.amx = max(self.input.max(), torch.tensor(0.0, dtype=self.dtype).to(self.device))
        self.amn = min(self.input.min(), torch.tensor(0.0, dtype=self.dtype).to(self.device))
        logger.info('the module type is {}'.format(self.module_type))
        logger.info('the data type is {}, the device is {}'.format(self.dtype, self.device))
        logger.info('the activation range is {:.2f}, {:.2f}'.format(self.amn, self.amx))
        logger.info('the weight range is {:.2f}, {:.2f}'.format(self.weight.min(), self.weight.max()))
        # calculate output
        self.output = self.get_output(self.input, self.weight)
        # prepare MinMax Observer for later Quantize
        self.aob = MinMaxObserver(self.a_qconfig.bit, self.a_qconfig.symmetric,
                                  self.a_qconfig.ch_axis).to(self.device).to(self.dtype)
        self.wob = MinMaxObserver(self.w_qconfig.bit, self.w_qconfig.symmetric,
                                  self.w_qconfig.ch_axis).to(self.device).to(self.dtype)

    def get_output(self, input, weight):
        if self.module_type == 'qkv':
            output = self.qkv_function(input, weight)
        elif self.module_type == 'up_and_gate':
            output = self.up_function(input, weight)
        elif self.module_type == 'down_proj':
            output = self.down_function(input, weight)
        else:
            raise NotImplementedError
        return output

    def quantize(self, X, observer, clipping_range=None):
        org_shape = X.shape
        if clipping_range is not None:
            if 'Group' in self.a_qconfig.quantizer:
                X = X.reshape(-1, self.a_qconfig.group_size)
                min_val_cur, max_val_cur = observer(X)
            elif 'Token' in self.a_qconfig.quantizer:
                X = X.reshape(-1, org_shape[-1])
                min_val_cur, max_val_cur = observer(X)
            else:
                min_val_cur, max_val_cur = clipping_range
        else:
            if 'Group' in self.w_qconfig.quantizer:
                X = X.reshape(-1, self.w_qconfig.group_size)
            min_val_cur, max_val_cur = observer(X)
        scale, zp = observer.calculate_qparams(min_val_cur, max_val_cur)
        if observer.ch_axis == -1:
            X_q = fake_quantize_per_tensor_affine(
                X, scale.item(), zp.item(),
                observer.quant_min, observer.quant_max)
        else:
            X_q = fake_quantize_per_channel_affine(
                X, scale, zp, observer.ch_axis,
                observer.quant_min, observer.quant_max)
        X_q = X_q.reshape(org_shape)
        return X_q

    def get_qoutput(self, input, weight, clipping_range=None):
        qinput = self.quantize(input, self.aob, clipping_range)
        qweight = self.quantize(weight, self.wob)
        return self.get_output(qinput, qweight)

    def cac_scale(self, min_range, max_range):
        mx_scale = torch.where(self.cmx > max_range, self.cmx / max_range, torch.tensor(1.0, dtype=self.dtype).to(self.device))
        mn_scale = torch.where(self.cmn < min_range, self.cmn / min_range, torch.tensor(1.0, dtype=self.dtype).to(self.device))
        final_scale = torch.max(mx_scale, mn_scale)
        return final_scale

    def get_best_scale(self, min_range, max_range):
        best_scale = self.cac_scale(min_range, max_range)
        logger.info('the best scale is {:.2f}, best min range is {:.2f}, \
            best max range is {:.2f}'.format(best_scale.max(), (self.input / best_scale).min(), (self.input / best_scale).max()))
        logger.info('the range of weight becomes {:.2f}, {:.2f}'.format((self.weight * best_scale).min(), (self.weight * best_scale).max()))
        return best_scale
        # return best_scale

    def loss_fx(self, pred, tgt, p=2.0):
        return (pred - tgt).abs().pow(p).sum(-1).mean()

    def cac_loss(self, min_range, max_range):
        cur_scale = self.cac_scale(min_range, max_range)
        qoutput = self.get_qoutput(self.input / cur_scale, self.weight * cur_scale, (min_range, max_range))
        return self.loss_fx(qoutput, self.output)

    def qkv_function(self, input, weight):
        B, N, C = input.shape
        head_dim = self.extra_dict['head_dim']
        qkv = torch.matmul(input, weight.T)
        sz_q = self.extra_dict['num_heads'] * head_dim
        sz_kv = self.extra_dict['num_key_value_heads'] * head_dim
        q = qkv[:, :, : sz_q].view(B, N, self.extra_dict['num_heads'], head_dim).transpose(1, 2)
        k = qkv[:, :, sz_q: sz_q + sz_kv].view(B, N, self.extra_dict['num_key_value_heads'], head_dim).transpose(1, 2)
        v = qkv[:, :, sz_q + sz_kv: ].view(B, N, self.extra_dict['num_key_value_heads'], head_dim).transpose(1, 2)

        kv_seq_len = k.shape[-2]
        cos, sin = self.extra_dict['cos_cached'][:, :, :kv_seq_len, ...], self.extra_dict['sin_cached'][:, :, :kv_seq_len, ...]
        q, k = quant_llama.apply_rotary_pos_emb(q, k, cos, sin, self.extra_dict['position_ids'])
        k = quant_llama.repeat_kv(k, self.extra_dict['num_key_value_groups'])
        v = quant_llama.repeat_kv(v, self.extra_dict['num_key_value_groups'])
        attn = (q / math.sqrt(head_dim)) @ k.transpose(-2, -1)
        attn = attn + self.extra_dict['attention_mask']
        attn = torch.max(attn, torch.tensor(torch.finfo(self.dtype).min))
        attn = attn.softmax(dim=-1, dtype=torch.float32).to(self.dtype)
        # bs, heads, token, token @ (bs, heads, token, dim)
        # bs, token, heads, dim
        # bs, heads, token, dim
        output = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return output[self.extra_dict['observation_mask'] == 1].to(torch.float32)

    def up_function(self, input, weight):
        B, N, _ = input.shape
        C, _ = weight.shape
        output = torch.matmul(input, weight.T).reshape(B, N, 2, C // 2).permute(2, 0, 1, 3)
        gate, up = output[0], output[1]
        output = self.extra_dict['act_fn'](gate) * up
        return output[self.extra_dict['observation_mask'] == 1].to(torch.float32)

    def down_function(self, input, weight):
        output = torch.matmul(input, weight.T)
        return output[self.extra_dict['observation_mask'] == 1].to(torch.float32)

    def forward(self,):
        pass


class Migrator1DRangeSearch(MigratorBase):

    def __init__(self, input, weight, a_qconfig, w_qconfig, module_type, extra_dict=None):
        super().__init__(input, weight, a_qconfig, w_qconfig, module_type, extra_dict)
        self.num = max(100, int(self.amx / 0.5))

    def cac_scale_loss(self, mn_range, mx_range):
        return self.cac_loss(torch.tensor(mn_range, dtype=self.dtype).to(self.device),
                             torch.tensor(mx_range, dtype=self.dtype).to(self.device))

    def search_migrate_range_1D(self,):
        best_loss = None
        bounds = (0.1, max(-self.amn.item(), self.amx.item()))
        step = (bounds[1] - bounds[0]) / self.num
        mn_range = -bounds[1]
        mx_range = bounds[1]
        st = bounds[1]
        cnt = 0
        while st >= bounds[0]:
            loss = self.cac_scale_loss(-st, st)
            if best_loss is None or best_loss > loss:
                best_loss = loss
                mn_range = -st
                mx_range = st
            cnt += 1
            if cnt % 10 == 0:
                logger.info('{:.2f} loss at iter {}'.format(loss, cnt))
            st -= step
        return (torch.tensor(mn_range, dtype=self.dtype).to(self.device),
                torch.tensor(mx_range, dtype=self.dtype).to(self.device))

    def forward(self,):
        best_range = self.search_migrate_range_1D()
        return self.get_best_scale(*best_range)

