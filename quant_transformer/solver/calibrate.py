import torch
import logging
import random
from datasets import load_dataset, load_from_disk
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


def prepare_data(tokenizer, calibrate_path, training_args):
    # raw_dataset = load_dataset("json", data_files=dataset_path, split="train")
    # random_rows = random.sample(range(raw_dataset.num_rows), calibrate)
    # raw_dataset = raw_dataset.select(random_rows)
    # raw_dataset.save_to_disk('/mnt/lustre/weixiuying/datasets/nlp_datasets/pile_cali')
    # calibrate_path = '/mnt/lustre/weixiuying/datasets/nlp_datasets/pile_cali'
    raw_dataset = load_from_disk(calibrate_path)

    def preprocess_function(examples):
        result = tokenizer(examples["text"], padding=True,
                           max_length=512, truncation=True)
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
        tmp['attention_mask'] = p['attention_mask'].to('cuda:0')
        tmp['input_ids'] = p['input_ids'].to('cuda:0')
        model(**tmp)
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
    # get cali data
    training_args = make_huggingface_training_args(batch_size)
    cali_data = prepare_data(lm.tokenizer, q_config.quant.calibrate_path, training_args)
    # get trainer
    dataloader = get_eval_dataloader(cali_data, training_args)
    fp_input, fp_output = prepare_input_output(lm.model, dataloader)
    from quant_transformer.quantization.quantized_module import QuantizedModule
    import time
    logger.info('begin migration!')
    st = time.time()
    enable_calibration_quantization(lm.model)
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
        else:
            import quant_transformer.quantization.migration as migration
        migration.fuse_migration(lm.model)
        lm.model.to(torch.float16)
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
