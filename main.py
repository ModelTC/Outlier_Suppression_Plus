import argparse
import json
import logging
import collections
import fnmatch
import yaml
import os
import numpy as np
from tqdm import tqdm
import torch
from easydict import EasyDict
import datasets

from lm_eval import tasks, evaluator
import lm_eval
from quant_transformer.model.quant_model import quantize_model
from quant_transformer.quantization.state import enable_quantization
from quant_transformer.solver.calibrate import calibrate
logger = logging.getLogger("OS+")


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        cur_config = config
        cur_path = config_file
        while 'root' in cur_config:
            root_path = os.path.dirname(cur_path)
            cur_path = os.path.join(root_path, cur_config['root'])
            with open(cur_path) as r:
                root_config = yaml.load(r, Loader=yaml.FullLoader)
                for k, v in root_config.items():
                    if k not in config:
                        config[k] = v
                cur_config = root_config
    config = EasyDict(config)
    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=MultiChoice(lm_eval.tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--dtype", type=str, default='float32')
    parser.add_argument("--is_quant", type=bool, default=False)
    parser.add_argument("--is_export", type=bool, default=False)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def load_model(model, model_args, dtype, q_config, args):
    if dtype == 'float16':
        dtype = torch.float16
    elif dtype == 'bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    if isinstance(model, str):
        if model_args is None:
            model_args = ""
        max_length = -1
        if hasattr(q_config, 'model') and hasattr(q_config.model, 'max_length'):
            max_length = q_config.model.max_length
        lm = lm_eval.models.get_model(model).create_from_arg_string(
            model_args, {"batch_size": args.batch_size, "device": args.device, 'dtype': dtype, 'max_length': max_length}
        )
        # load quant models
        if args.is_quant:
            lm.model = quantize_model(lm.model, q_config)
        lm.prepare_for_inference()
    else:
        assert isinstance(model, lm_eval.base.LM)
        lm = model
    if not args.no_cache:
        lm = lm_eval.base.CachingLM(
            lm,
            "lm_cache/"
            + model
            + "_"
            + model_args.replace("=", "-").replace(",", "_").replace("/", "-")
            + ".db",
        )
        lm = lm.lm
    return lm


@torch.no_grad()
def main():
    args = parse_args()
    if args.config:
        q_config = parse_config(args.config)
        if args.is_export:
            args.save_path = '/'.join(args.config.split('/')[: -1])
    else:
        q_config = None
    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = lm_eval.tasks.ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), lm_eval.tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)


    lm = load_model(
        model=args.model,
        model_args=args.model_args,
        dtype=args.dtype,
        q_config=q_config,
        args=args)

    if args.is_export:
        shift_list, scale_list = calibrate(lm, args.batch_size, q_config, True)
        for i in range(len(shift_list)):
            shift_list[i].cpu()
            scale_list[i].cpu()
        print(args.save_path)
        torch.save({
            'shift_list': shift_list,
            'scale_list': scale_list,
        }, os.path.join(args.save_path, 'scale_shift_list.pth'))
        return

    if args.is_quant:
        calibrate(lm, args.batch_size, q_config)
        enable_quantization(lm.model, except_quantizer=getattr(q_config.quant, 'except_quantizer', None))

    from transformers import AutoTokenizer
    for task in task_names:
        if task in ["wikitext",]:
            test_data = datasets.load_from_disk("/mnt/cache/weixiuying.vendor/wikitext/wikitext_test")
            test_enc = lm.tokenizer("\n\n".join(test_data["text"]), return_tensors="pt")
            test_enc = test_enc.input_ids
            nsamples = test_enc.numel() // lm.max_length
            nlls = []
            for i in tqdm(range(nsamples)):
                batched_inps = test_enc[:, (i * lm.max_length) : ((i + 1) * lm.max_length)].to('cuda:0')
                batched_labels = test_enc[:, (i * lm.max_length) : ((i + 1) * lm.max_length)].to(lm.model.lm_head.weight.device)
                loss = lm.model(batched_inps, labels=batched_labels).loss
                neg_log_likelihood = loss.float() * lm.max_length
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.max_length))

            results = collections.defaultdict(dict)
            versions = collections.defaultdict(dict)
            results[task]['ppl'] = ppl.item()
            versions[task] = 0
            results = {"results": dict(results), "versions": dict(versions)}
        else:
            results = lm_eval.evaluator.simple_evaluate(
                model=args.model,
                model_args=args.model_args,
                tasks=[task,],
                num_fewshot=args.num_fewshot,
                batch_size=args.batch_size,
                device=args.device,
                no_cache=args.no_cache,
                limit=args.limit,
                description_dict=description_dict,
                decontamination_ngrams_path=args.decontamination_ngrams_path,
                check_integrity=args.check_integrity,
                lm=lm,
            )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
    )
    print(lm_eval.evaluator.make_table(results))


if __name__ == "__main__":
    main()
