import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from accelerate.big_modeling import dispatch_model, infer_auto_device_map, get_balanced_memory


def deploy_bloom(scale_list, shift_list):
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            # attn ln
            attn_ln.bias.data -= shift_list[cnt].to(attn_ln.bias.data.device)
            attn_ln.weight.data /= scale_list[cnt].to(attn_ln.bias.data.device)
            attn_ln.bias.data /= scale_list[cnt].to(attn_ln.bias.data.device)
            # qkv
            qkv.bias.data += shift_list[cnt].to(qkv.weight.data.device) @ qkv.weight.data.T
            qkv.weight.data *= scale_list[cnt].to(qkv.weight.data.device)
            cnt += 1
            
            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            # ffn ln
            ffn_ln.bias.data -= shift_list[cnt].to(ffn_ln.bias.data.device)
            ffn_ln.weight.data /= scale_list[cnt].to(ffn_ln.bias.data.device)
            ffn_ln.bias.data /= scale_list[cnt].to(ffn_ln.bias.data.device)
            # fc1
            fc1.bias.data += shift_list[cnt].to(fc1.weight.data.device) @ fc1.weight.data.T
            fc1.weight.data *= scale_list[cnt].to(fc1.weight.data.device)
            cnt += 1
    return model


def deploy_opt(scale_list, shift_list):
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            q, k, v = module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj
            # attn ln
            attn_ln.bias.data -= shift_list[cnt].to(attn_ln.bias.data.device)
            attn_ln.weight.data /= scale_list[cnt].to(attn_ln.bias.data.device)
            attn_ln.bias.data /= scale_list[cnt].to(attn_ln.bias.data.device)
            # qkv
            q.bias.data += shift_list[cnt].to(q.weight.data.device) @ q.weight.data.T
            q.weight.data *= scale_list[cnt].to(q.weight.data.device)
            k.bias.data += shift_list[cnt].to(k.weight.data.device) @ k.weight.data.T
            k.weight.data *= scale_list[cnt].to(k.weight.data.device)
            v.bias.data += shift_list[cnt].to(v.weight.data.device) @ v.weight.data.T
            v.weight.data *= scale_list[cnt].to(v.weight.data.device)
            cnt += 1
            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            # ffn ln
            ffn_ln.bias.data -= shift_list[cnt].to(ffn_ln.bias.data.device)
            ffn_ln.weight.data /= scale_list[cnt].to(ffn_ln.bias.data.device)
            ffn_ln.bias.data /= scale_list[cnt].to(ffn_ln.bias.data.device)
            # fc1
            fc1.bias.data += shift_list[cnt].to(fc1.weight.data.device) @ fc1.weight.data.T
            fc1.weight.data *= scale_list[cnt].to(fc1.weight.data.device)
            cnt += 1
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--scale_shift_list", required=True)
    parser.add_argument("--model_type", default="opt")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
    ).eval()
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=model._no_split_modules,
        dtype=torch.float16
    )
    device_map = infer_auto_device_map(
        model,
        no_split_module_classes=model._no_split_modules,
        dtype=torch.float16,
        max_memory=max_memory,
    )
    dispatch_model(model, device_map=device_map)
    scale_shift_list = torch.load(args.scale_shift_list)
    scale_list = scale_shift_list['scale_list']
    shift_list = scale_shift_list['shift_list']
    if args.model_type == 'opt':
        model = deploy_opt(scale_list, shift_list)
        model.to(torch.float16)
        model.cpu()
        model.model.save_pretrained(args.output_path, max_shard_size="5GB", safe_serialization=True)
    elif args.model_type == 'bloom':
        model = deploy_bloom(scale_list, shift_list)
        model.to(torch.float16)
        model.cpu()
        model.transformer.save_pretrained(args.output_path, max_shard_size="5GB", safe_serialization=True)
    else:
        raise NotImplementedError
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        revision="main",
        use_fast=False,
    )
    tokenizer.save_pretrained(args.output_path)
