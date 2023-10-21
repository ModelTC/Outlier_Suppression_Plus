import torch
import logging
import random
import os
from datasets import load_dataset, load_from_disk, Dataset
from torch.utils.data import DataLoader
from transformers import TrainingArguments
from quant_transformer.quantization.state import enable_calibration_woquantization, enable_calibration_quantization, disable_all
logger = logging.getLogger("OS+")
# dataset_path = '/mnt/lustre/weixiuying/datasets/nlp_datasets/pile/val.jsonl.zst'


def make_huggingface_training_args(batch_size):
    training_args = TrainingArguments(
        output_dir="output_dir",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
    )
    return training_args


def prepare_data(tokenizer, calibrate_path, training_args, calibrate_num, max_length):
    if 'wiki' in calibrate_path:
        if 'cali' not in calibrate_path:
            raw_dataset = load_from_disk(calibrate_path)
            train_data = tokenizer("\n\n".join(raw_dataset["text"]), return_tensors="pt")
            cali_data = []
            for _ in range(calibrate_num):
                i = random.randint(0, train_data.input_ids.shape[1] - max_length - 1)
                j = i + max_length
                inp = train_data.input_ids[:, i: j][0]
                cali_data.append(inp)
            cali_data = Dataset.from_dict({"input_ids": cali_data})
            cali_data.set_format(type='torch', columns=['input_ids'])
            calibrate_path = os.path.join('/'.join(calibrate_path.split('/')[: -1]), 'wiki_cali')
            cali_data.save_to_disk(calibrate_path)
        cali_data = load_from_disk(calibrate_path)
    elif 'pile' in calibrate_path:
        if 'cali' not in calibrate_path:
            raw_dataset = load_dataset("json", data_files=calibrate_path, split="train")
            random_rows = random.sample(range(raw_dataset.num_rows), calibrate)
            raw_dataset = raw_dataset.select(random_rows)
            calibrate_path = os.path.join('/'.join(calibrate_path.split('/')[: -1]), 'pile_cali')
            raw_dataset.save_to_disk(calibrate_path)
        raw_dataset = load_from_disk(calibrate_path)
        def preprocess_function(examples):
            result = tokenizer(examples["text"], padding=True,
                            max_length=max_length, truncation=True)
            return result
        with training_args.main_process_first(desc="dataset map pre-processing"):
            cali_data = raw_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=True,
                desc="Running tokenizer on dataset",
            )
        cali_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        cali_data.remove_columns(['meta', 'text'])
    return cali_data


def prepare_input_output(model, dataloader):
    logger.info('**prepare fp input and output**')
    fp_input, fp_output = [], []
    for p in dataloader:
        tmp = {}
        if 'attention_mask' in p:
            tmp['attention_mask'] = p['attention_mask'].to('cuda:0')
        tmp['input_ids'] = p['input_ids'].to('cuda:0')
        # output = model(**tmp)[0].to(torch.float32).cpu()
        fp_input.append(tmp)
        # fp_output.append(output[tmp['attention_mask'] == 1, :])
    return fp_input, fp_output


def calibrate_batch(model, fp_input):
    logger.info("*** Calibrate ***")
    for batch in fp_input:
        model(**batch)


def get_eval_dataloader(eval_dataset, training_args):

    return DataLoader(
        eval_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        drop_last=training_args.dataloader_drop_last,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
    )


@torch.no_grad()
def calibrate(lm, batch_size, q_config, is_export=False):
    logger.info('***** Calibration *****')
    logger.info('the quantization config is {}'.format(q_config))
    # get cali data
    training_args = make_huggingface_training_args(batch_size)
    cali_data = prepare_data(lm.tokenizer, q_config.quant.calibrate_path, training_args, q_config.quant.calibrate, q_config.model.max_length)
    # get trainer
    dataloader = get_eval_dataloader(cali_data, training_args)
    fp_input, fp_output = prepare_input_output(lm.model, dataloader)
    from quant_transformer.quantization.quantized_module import QuantizedModule
    import time
    logger.info('begin migration!')
    st = time.time()
    enable_calibration_quantization(lm.model, except_quantizer=getattr(q_config.quant, 'except_quantizer', None))
    if hasattr(q_config.quant, 'migrate') and q_config.quant.migrate:
        for name, module in lm.model.named_modules():
            if isinstance(module, QuantizedModule):
                module.set_cac_migrate(True)
        calibrate_batch(lm.model, [fp_input[0]])
        for name, module in lm.model.named_modules():
            if isinstance(module, QuantizedModule):
                module.set_cac_migrate(False)
        if lm.model.__class__.__name__ == 'QuantizedBloomForCausalLM':
            import quant_transformer.quantization.migration_bloom as migration
        elif lm.model.__class__.__name__ == 'QuantizedLlamaForCausalLM':
            import quant_transformer.quantization.migration_llama as migration
        else:
            import quant_transformer.quantization.migration as migration
        migration.fuse_migration(lm.model)
    ed = time.time()
    logger.info('cost {:.4f} time'.format(ed - st))
    if is_export:
        return migration.shift_list, migration.scale_list
    # quantize activation
    if q_config.quant.a_qconfig.observer == "AvgTokenQuantileObserver":
        from .token_wise_clipping import token_wise_clipping
        st = time.time()
        # quantize activation
        disable_all(lm.model)
        token_wise_clipping(lm.model, fp_input, fp_output, q_config, batch_size)
        ed = time.time()
        logger.info('cost {:.4f} time'.format(ed - st))
    else:
        calibrate_batch(lm.model, fp_input)
