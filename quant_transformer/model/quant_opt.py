"""PyTorch Quantized OPT model."""
import random
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import LayerNorm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import logging

from transformers.modeling_utils import (
    GenerationMixin,
    ModuleUtilsMixin,
    PreTrainedModel
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
    SequenceClassifierOutputWithPast
)

from quant_transformer.quantization import Quantizer, QuantizedLayer, QuantizedModule
from quant_transformer.quantization.migration import migration
from quant_transformer.model.util_layernorm import QuantizedLayerNorm, Identity
logger = logging.getLogger("OS+")


def _make_causal_mask(input_ids_shape, dtype, past_key_values_length=0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask, dtype, tgt_len=None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class QuantizedOPTAttention(QuantizedModule):

    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend='academic'):
        super().__init__(backend=backend)
        self.qinput = qinput
        self.embed_dim = org_module.embed_dim
        self.num_heads = org_module.num_heads
        self.dropout = org_module.dropout
        self.head_dim = org_module.head_dim

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = org_module.scaling
        self.is_decoder = org_module.is_decoder

        self.k_proj = QuantizedLayer(org_module.k_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.v_proj = QuantizedLayer(org_module.v_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.q_proj = QuantizedLayer(org_module.q_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.out_proj = QuantizedLayer(org_module.out_proj, None, w_qconfig, a_qconfig, True)

        # for torch.bmm
        self.query_permute_post_act_fake_quant = Quantizer(None, a_qconfig)
        self.key_transpose_post_act_fake_quant = Quantizer(None, a_qconfig)
        self.value_permute_post_act_fake_quant = Quantizer(None, a_qconfig)
        self.attention_probs_post_act_fake_quant = Quantizer(None, a_qconfig)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
        observation_mask=None
    ):
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()
        # get query proj
        query_states = self.query_permute_post_act_fake_quant(
            self.q_proj(hidden_states) * self.scaling, observation_mask, 1
        )
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(
                self.key_transpose_post_act_fake_quant(self.k_proj(key_value_states), observation_mask, 1),
                -1,
                bsz
            )
            value_states = self._shape(
                self.value_permute_post_act_fake_quant(self.v_proj(key_value_states), observation_mask, 1),
                -1,
                bsz
            )
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(
                self.key_transpose_post_act_fake_quant(self.k_proj(hidden_states), observation_mask, 1),
                -1,
                bsz
            )
            value_states = self._shape(
                self.value_permute_post_act_fake_quant(self.v_proj(hidden_states), observation_mask, 1),
                -1,
                bsz
            )
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(
                self.key_transpose_post_act_fake_quant(self.k_proj(hidden_states), observation_mask, 1),
                -1,
                bsz
            )
            value_states = self._shape(
                self.value_permute_post_act_fake_quant(self.v_proj(hidden_states), observation_mask, 1),
                -1,
                bsz
            )

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_probs = self.attention_probs_post_act_fake_quant(attn_probs)
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output, observation_mask, 1)

        return attn_output, attn_weights_reshaped, past_key_value


class QuantizedOPTDecoderLayer(QuantizedModule):

    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend='academic'):
        super().__init__(backend)
        self.first_layer_norm = QuantizedLayerNorm(org_module.self_attn_layer_norm)
        self.second_layer_norm = QuantizedLayerNorm(org_module.final_layer_norm)

        self.qinput = qinput
        self.w_qconfig = w_qconfig
        self.a_qconfig = a_qconfig
        self.embed_dim = org_module.embed_dim
        self.do_layer_norm_before = org_module.do_layer_norm_before
        if self.qinput:
            self.first_act_fake_quant = Quantizer(None, a_qconfig)
        self.first_identity = Identity()
        self.self_attn = QuantizedOPTAttention(
            org_module.self_attn,
            w_qconfig,
            a_qconfig,
            qinput=False,
            backend=self.backend
        )
        self.dropout = org_module.dropout
        self.second_act_fake_quant = Quantizer(None, a_qconfig)
        self.second_identity = Identity()
        self.fc1_act = QuantizedLayer(org_module.fc1, org_module.activation_fn,
                                      w_qconfig, a_qconfig, qinput=False)
        self.fc2 = QuantizedLayer(org_module.fc2, None, w_qconfig, a_qconfig, True)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
        use_cache=False,
        past_key_value=None,
        observation_mask=None
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        # layernorm, gamma migration, act_fake_quant
        if self.do_layer_norm_before:
            residual = self.first_identity(hidden_states)
            hidden_states = self.first_layer_norm(hidden_states)
        # migration
        if self.cac_migrate:
            cmx = hidden_states.max(0)[0].max(0)[0]
            cmn = hidden_states.min(0)[0].min(0)[0]
            logger.info('the original min range is {}, the original max range is {}'.format(hidden_states.min(), hidden_states.max()))

            shift = (cmx + cmn) / 2
            # bias
            hidden_states -= shift
            # self.first_layer_norm.bias.data -= shift
            self.self_attn.q_proj.module.bias.data += shift @ self.self_attn.q_proj.module.weight.data.T
            self.self_attn.k_proj.module.bias.data += shift @ self.self_attn.k_proj.module.weight.data.T
            self.self_attn.v_proj.module.bias.data += shift @ self.self_attn.v_proj.module.weight.data.T
            if not self.do_layer_norm_before:
                self.first_identity.set_migrate_bias(shift)
            # calculate scale
            weight_list = torch.cat([self.self_attn.q_proj.module.weight,
                                     self.self_attn.k_proj.module.weight,
                                     self.self_attn.v_proj.module.weight])
            extra_dict = {
                'bias': torch.cat([self.self_attn.q_proj.module.bias,
                                  self.self_attn.k_proj.module.bias,
                                  self.self_attn.v_proj.module.bias]),
                'num_heads': self.self_attn.num_heads,
                'scaling': self.self_attn.scaling,
                'attention_mask': attention_mask,
                'observation_mask': observation_mask,
                'shift': shift
            }
            # update scale
            best_scale = \
                migration(hidden_states, weight_list, self.a_qconfig, self.w_qconfig, 'qkv', extra_dict)
            hidden_states /= best_scale
            self.self_attn.q_proj.module.weight.data *= best_scale
            self.self_attn.k_proj.module.weight.data *= best_scale
            self.self_attn.v_proj.module.weight.data *= best_scale
            if not self.do_layer_norm_before:
                self.first_identity.set_migrate_scale(best_scale)

        hidden_states = self.first_act_fake_quant(hidden_states, observation_mask, 1)

        if not self.do_layer_norm_before:
            residual = self.first_identity(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            observation_mask=observation_mask,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.first_layer_norm(hidden_states)

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            residual = self.second_identity(hidden_states)
            hidden_states = self.second_layer_norm(hidden_states)
        # migration
        if self.cac_migrate:
            cmx = hidden_states.max(0)[0].max(0)[0]
            cmn = hidden_states.min(0)[0].min(0)[0]
            logger.info('the original min range is {}, the original max range is {}'.format(hidden_states.min(), hidden_states.max()))

            # shift
            shift = (cmx + cmn) / 2
            hidden_states -= shift
            self.fc1_act.module.bias.data += shift @ self.fc1_act.module.weight.data.T
            if not self.do_layer_norm_before:
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
            if not self.do_layer_norm_before:
                self.second_identity.set_migrate_scale(best_scale)
        hidden_states = self.second_act_fake_quant(hidden_states, observation_mask, 1)
        if not self.do_layer_norm_before:
            residual = self.second_identity(hidden_states)
        # Fully Connected
        hidden_states = self.fc1_act(hidden_states, observation_mask, 1)
        hidden_states = self.fc2(hidden_states, observation_mask, 1)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.second_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class QuantizedOPTDecoder(QuantizedModule):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OPTDecoderLayer`]
    """

    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend='academic'):
        super().__init__(backend)
        self.qinput = qinput
        self.config = org_module.config
        self.dropout = org_module.dropout
        self.layerdrop = org_module.layerdrop
        self.padding_idx = org_module.padding_idx
        self.max_target_positions = org_module.max_target_positions
        self.vocab_size = org_module.vocab_size

        self.embed_tokens = org_module.embed_tokens
        self.embed_positions = org_module.embed_positions
        if self.config.word_embed_proj_dim != self.config.hidden_size:
            self.project_in = QuantizedLayer(org_module.project_in, None, w_qconfig, a_qconfig, True)
            logger.info(self.project_in)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164

        self.layers = nn.ModuleList()
        for i in range(self.config.num_hidden_layers):
            self.layers.append(
                QuantizedOPTDecoderLayer(
                    org_module.layers[i],
                    w_qconfig,
                    a_qconfig,
                    qinput=True
                )
            )

        self.gradient_checkpointing = org_module.gradient_checkpointing

        self.last_layer_norm = org_module.final_layer_norm
        if self.config.word_embed_proj_dim != self.config.hidden_size:
            self.project_out = QuantizedLayer(
                org_module.project_out, None, w_qconfig, a_qconfig, qinput=True)
        else:
            self.project_out = None

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(inputs_embeds.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        observation_mask=None
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
                Indices can be obtained using [`OPTTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.
                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
            observation_mask = attention_mask.clone()
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, False, past_key_value, observation_mask)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    observation_mask=observation_mask
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.last_layer_norm is not None:
            hidden_states = self.last_layer_norm(hidden_states)
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states, observation_mask, 1)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class QuantizedOPTModel(QuantizedModule, ModuleUtilsMixin):

    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend='academic'):
        super().__init__(backend=backend)
        self.qinput = qinput
        self.decoder = QuantizedOPTDecoder(
            org_module.decoder,
            w_qconfig,
            a_qconfig,
            qinput=self.qinput,
            backend=self.backend
        )

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        observation_mask=None
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            observation_mask=observation_mask
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


class QuantizedOPTForCausalLM(QuantizedModule, GenerationMixin, ModuleUtilsMixin):

    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend='academic', is_remove_padding=False):
        super().__init__(backend)
        self._no_split_modules = ["QuantizedOPTDecoderLayer", "QuantizedOPTLearnedPositionalEmbedding", "QuantizedLayer", "QuantizedModule"]
        self.qinput = qinput
        self.config = org_module.config
        self.model = QuantizedOPTModel(
            org_module.model,
            w_qconfig,
            a_qconfig,
            self.qinput,
            backend=self.backend
        )
        self.lm_head = org_module.lm_head
        self.is_remove_padding = is_remove_padding

    def tie_weights(self):
        self.get_output_embeddings().weight = self.get_input_embeddings().weight

    def is_remove_padding(self, is_remove_padding=False):
        self.is_remove_padding = is_remove_padding

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
                Indices can be obtained using [`OPTTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.
                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.
                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        Returns:
        Example:
        ```python
        >>> from transformers import GPT2Tokenizer, OPTForCausalLM
        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")
        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.is_remove_padding and attention_mask is not None:
            observation_mask = attention_mask.clone()
        else:
            observation_mask = None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            observation_mask=observation_mask
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, use_cache=None, **kwargs):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past:
            input_ids = input_ids[:, -1:]
        # first step, decoder_cached_states are empty
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
