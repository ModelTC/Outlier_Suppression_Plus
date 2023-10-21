"""PyTorch Quantized BLOOM model."""
import math
import random
import warnings
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import Optional, Tuple, Union

from transformers.modeling_utils import (
    GenerationMixin,
    ModuleUtilsMixin,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)

import logging
from quant_transformer.quantization import Quantizer, QuantizedLayer, QuantizedModule
from quant_transformer.quantization.migration_bloom import migration
from quant_transformer.model.util_layernorm import QuantizedLayerNorm, Identity
logger = logging.getLogger("OS+")


def _make_causal_mask(
    input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = torch.empty((target_length, target_length + past_key_values_length), dtype=torch.bool, device=device)
    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    seq_ids = torch.arange(target_length, device=device)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
    return expanded_mask


def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)


def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.
    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)


class QuantizedBloomAttention(QuantizedModule):

    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend='academic'):
        super().__init__(backend=backend)
        self.qinput = qinput
        self.hidden_size = org_module.hidden_size
        self.num_heads = org_module.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0
        self.query_key_value = QuantizedLayer(org_module.query_key_value, None, w_qconfig, a_qconfig, self.qinput)
        self.dense = QuantizedLayer(org_module.dense, None, w_qconfig, a_qconfig, True)

        # activation quantizer
        self.query_post_act_fake_quant = Quantizer(None, a_qconfig)
        self.key_post_act_fake_quant = Quantizer(None, a_qconfig)
        self.value_post_act_fake_quant = Quantizer(None, a_qconfig)
        self.attention_probs_post_act_fake_quant = Quantizer(None, a_qconfig)

    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
        storage as `fused_qkv`
        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]
        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads together over the last dimenstion
        Args:
            x: (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]
        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        observation_mask=None,
    ):
        fused_qkv = self.query_key_value(hidden_states, observation_mask, 1)  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
        query_layer = self.query_post_act_fake_quant(query_layer, observation_mask, 1)
        key_layer = self.key_post_act_fake_quant(key_layer, observation_mask, 1)
        value_layer = self.value_post_act_fake_quant(value_layer, observation_mask, 1)

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=2)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, _, kv_length = key_layer.shape

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        # [batch_size * num_heads, q_length, kv_length]
        # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
        matmul_result = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        # change view to [batch_size, num_heads, q_length, kv_length]
        assert q_length == kv_length
        attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16:
            attention_scores = attention_scores.to(torch.float)
        attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)

        # # [batch_size, num_heads, q_length, kv_length]
        # attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)
        # attention_probs
        attention_probs_reshaped = self.attention_probs_post_act_fake_quant(
            attention_probs_reshaped
        )
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = self._merge_heads(context_layer)
        output_tensor = self.dense(context_layer, observation_mask, 1)

        output_tensor = output_tensor + residual

        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs


class QuantizedBloomBlock(QuantizedModule):

    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend='academic'):
        super().__init__(backend=backend)
        self.w_qconfig = w_qconfig
        self.a_qconfig = a_qconfig
        self.qinput = qinput
        self.first_layer_norm = QuantizedLayerNorm(org_module.input_layernorm)
        self.second_layer_norm = QuantizedLayerNorm(org_module.post_attention_layernorm)
        if self.qinput:
            self.first_act_fake_quant = Quantizer(None, a_qconfig)
        self.second_act_fake_quant = Quantizer(None, a_qconfig)
        self.post_ln = org_module.apply_residual_connection_post_layernorm
        self.first_identity = Identity()
        self.second_identity = Identity()

        self.self_attn = QuantizedBloomAttention(
            org_module.self_attention,
            w_qconfig,
            a_qconfig,
            qinput=False,
        )
        self.fc1_act = QuantizedLayer(org_module.mlp.dense_h_to_4h, org_module.mlp.gelu_impl,
                                      w_qconfig, a_qconfig, qinput=False)
        self.fc2 = QuantizedLayer(org_module.mlp.dense_4h_to_h, None, w_qconfig, a_qconfig, True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        observation_mask=None,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]

        if not self.post_ln:
            residual = self.first_identity(hidden_states)
        hidden_states = self.first_layer_norm(hidden_states)

        if self.cac_migrate:
            cmx = hidden_states.max(0)[0].max(0)[0]
            cmn = hidden_states.min(0)[0].min(0)[0]
            logger.info('the original min range is {}, the original max range is {}'.format(hidden_states.min(), hidden_states.max()))

            shift = (cmx + cmn) / 2
            # bias
            hidden_states -= shift
            # self.first_layer_norm.bias.data -= shift
            self.self_attn.query_key_value.module.bias.data += shift @ self.self_attn.query_key_value.module.weight.data.T
            if self.post_ln:
                self.first_identity.set_migrate_bias(shift)
            # calculate scale
            weight_list = self.self_attn.query_key_value.module.weight.data
            extra_dict = {
                'bias': self.self_attn.query_key_value.module.bias.data,
                'num_heads': self.self_attn.num_heads,
                'head_dim': self.self_attn.head_dim,
                'alibi': alibi,
                'scaling': self.self_attn.inv_norm_factor,
                'attention_mask': attention_mask,
                'observation_mask': observation_mask,
                'shift': shift
            }
            # update scale
            best_scale = \
                migration(hidden_states, weight_list, self.a_qconfig, self.w_qconfig, 'qkv', extra_dict)
            hidden_states /= best_scale
            self.self_attn.query_key_value.module.weight.data *= best_scale
            if self.post_ln:
                self.first_identity.set_migrate_scale(best_scale)
        hidden_states = self.first_act_fake_quant(hidden_states, observation_mask, 1)

        if self.post_ln:
            residual = self.first_identity(hidden_states)
        # Self attention.
        attn_outputs = self.self_attn(
            hidden_states,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            observation_mask=observation_mask,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]
        if not self.post_ln:
            residual = self.second_identity(attention_output)
        hidden_states = self.second_layer_norm(attention_output)
        if self.cac_migrate:
            cmx = hidden_states.max(0)[0].max(0)[0]
            cmn = hidden_states.min(0)[0].min(0)[0]
            # shift
            logger.info('the original min range is {}, the original max range is {}'.format(hidden_states.min(), hidden_states.max()))

            shift = (cmx + cmn) / 2
            hidden_states -= shift
            self.fc1_act.module.bias.data += shift @ self.fc1_act.module.weight.data.T
            if self.post_ln:
                self.second_identity.set_migrate_bias(shift)
            # calculate scale
            extra_dict = {
                'observation_mask': observation_mask,
                'shift': shift
            }
            best_scale = \
                migration(hidden_states, self.fc1_act.module.weight, self.a_qconfig, self.w_qconfig, 'fc1', extra_dict)
            # update scale
            hidden_states /= best_scale
            self.fc1_act.module.weight.data *= best_scale
            if self.post_ln:
                self.second_identity.set_migrate_scale(best_scale)

        hidden_states = self.second_act_fake_quant(hidden_states, observation_mask, 1)
        if self.post_ln:
            residual = self.second_identity(hidden_states)
        # Fully Connected
        hidden_states = self.fc1_act(hidden_states, observation_mask, 1)
        hidden_states = self.fc2(hidden_states, observation_mask, 1)
        output = residual + hidden_states

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions


class QuantizedBloomModel(QuantizedModule, ModuleUtilsMixin):

    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend='academic'):
        super().__init__(backend=backend)
        self.config = org_module.config
        self.embed_dim = org_module.embed_dim
        self.num_heads = org_module.num_heads

        # Embedding + LN Embedding
        self.word_embeddings = org_module.word_embeddings
        self.word_embeddings_layernorm = org_module.word_embeddings_layernorm

        # Transformer blocks
        self.h = nn.ModuleList(
            [QuantizedBloomBlock(block, w_qconfig, a_qconfig, qinput=True) for block in org_module.h]
        )

        # Final Layer Norm
        self.ln_f = org_module.ln_f

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.word_embeddings

    def _prepare_attn_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
    ) -> torch.BoolTensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, device=device, past_key_values_length=past_key_values_length
            )

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        observation_mask=None,
        **deprecated_arguments
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
            observation_mask = attention_mask.clone()
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=causal_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                alibi=alibi,
                observation_mask=observation_mask,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class QuantizedBloomForCausalLM(QuantizedModule, GenerationMixin):

    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend='academic', is_remove_padding=True):
        super().__init__(backend=backend)
        self._no_split_modules = ["QuantizedBloomBlock", "QuantizedLayer", "QuantizedModule"]
        self.transformer = QuantizedBloomModel(
            org_module.transformer,
            w_qconfig,
            a_qconfig,
            qinput=True,
        )
        self.config = org_module.config
        self.lm_head = org_module.lm_head
        self.is_remove_padding = is_remove_padding

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> dict:
        # only last token for input_ids if past is not None
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

            # the cache may be in the stardard format (e.g. in contrastive search), convert to bloom's format if needed
            if past[0][0].shape[0] == input_ids.shape[0]:
                past = self._convert_to_bloom_cache(past)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        observation_mask=None,
        **deprecated_arguments
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.is_remove_padding and attention_mask is not None:
            observation_mask = attention_mask.clone()
        else:
            observation_mask = None

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            observation_mask=observation_mask,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _reorder_cache(
        self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        Output shares the same memory storage as `past`.
        """
        standardized_past = self._convert_to_standard_cache(past, batch_size=len(beam_idx))

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        return self._convert_to_bloom_cache(reordered_past)
