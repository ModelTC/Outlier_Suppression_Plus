# Outlier Suppression+
Official PyTorch implementation of  <a href="https://arxiv.org/abs/2304.09145">Outlier Suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling</a>

## Overview

The outlier suppression+ effectively suppresses outliers in large language models for better quantization performance without extra inference burden. To achieve this, it adopts a shifting and scaling operation which first eliminates the asymmetric property of outliers then scales down those concentrated outliers on certain channels (We put detailed analysis of outliers in [outlier_analysis.md](outlier_analysis.md)). 


It now supports OPT, BLOOM, and BLOOMZ, and can be applied to other models. As the method only changes the floating-point model such as weights and biases, we can easily export a new FP model with weaker outliers, enjoying convenience for further development.
<p align="center">
  <img src="figure/outlier_suppression_plus.png">
</p>
<!-- ![](outlier_suppression_plus.png =100x100) -->

## Usage
### Preparation
* Prepare A100 to run the code.
* Install transformers (version 4.23.1) and datasets (version 2.7.1). 
* Prepare your own OPTs or BLOOMs models.
* For codes of FP tasks, we adopt the implementation in <a href="https://github.com/EleutherAI/lm-evaluation-harness">lm-evaluation-harness</a>. Thus, download your dataset and update your own data path in lm_eval/dataset directory for later evaluation.
* Considering quantization on zero-shot tasks here, additionally prepare the pile datasets (part of train datasets) for later calibration. Please first download the PILE dataset to ${pile_path}, and run the following code which will randomly select 128 samples for calibration. 

    ```
    dataset_path = ${pile_path}
    calibration_size = 128
    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")
    random_rows = random.sample(range(raw_dataset.num_rows), calibration_size)
    raw_dataset = raw_dataset.select(random_rows)
    raw_dataset.save_to_disk(${pile_cali_path})
    ```

### Evaluation
This supports OPTs, BLOOMs, BLOOMZ models. We consider two kinds of settings for static quantization. One is symmetric and per-tensor quantization which represents the fast speed. Another one is asymmetric and per-channel (weight) quantization which represents higher accuracy.

Here, we give the config for each task. Directory int8_opt_tensor means accuracy of "symmetric and per-tensor quantization" which is marked with * in the paper.

```
exp
├── int8_opt
├── int8_opt_tensor
|   └──q_config.yaml
├── int6_opt
├── int8_bloom_tensor
├── int8_bloomz_tensor
├── int6_bloomz
└── int6_bloom
```

Then, run with the following commands. Here is the command for OPTs.
```
# opt.sh
model_size=66b
task_name=winogrande
model_path=model_path 
quant_dir=exp/int8_opt_tensor # q_config.yaml path
is_quant=True # True: quantization; False: fp16

export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py \
    --model opt \
    --model_args pretrained=$model_path \
    --tasks $task_name \
    --batch_size 16 \
    --no_cache \
    --is_fp16 \
    --is_quant ${is_quant} \
    --config ${quant_dir}/q_config.yaml \
    2>&1 | tee ${quant_dir}/${model_size}_${task_name}.log
```
Here is the command for BLOOM and BLOOMZ.
```
# bloom.sh
model_size=176b
task_name=winogrande
model_path=model_path 
quant_dir=exp/int8_bloom_tensor # q_config.yaml path
is_quant=True # True: quantization; False: fp16

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python main.py \
    --model opt \
    --model_args pretrained=$model_path \
    --tasks $task_name \
    --batch_size 12 \
    --no_cache \
    --is_fp16 \
    --is_quant ${is_quant} \
    --config ${quant_dir}/q_config.yaml \
    2>&1 | tee ${quant_dir}/${model_size}_${task_name}.log
```

It will first calculate the appropriate shifting and scaling values, then migrate them to later modules and fuse them in previous operations. 

### Export for deployment
As the method only changes the floating-point model such as weights and biases, we can easily export a new FP model with weaker outliers, enjoying convenience for further development.
* Obtain the scaling and shifting values by turning on the "is_export" choice in opt.sh file. It will store the calculated values in the ${quant_dir}.
```
# opt.sh
model_size=6.7b
task_name=winogrande
model_path=model_path 
quant_dir=exp/int8_opt_tensor # q_config.yaml path
is_quant=True # True: quantization; False: fp16
export CUDA_VISIBLE_DEVICES=0
python main.py \
    --model opt \
    --model_args pretrained=$model_path \
    --tasks $task_name \
    --batch_size 16 \
    --no_cache \
    --is_fp16 \
    --is_quant ${is_quant} \
    --is_export \
    --config ${quant_dir}/q_config.yaml \
    2>&1 | tee ${quant_dir}/${model_size}_${task_name}.log
```
* Use them to update the original FP model by running export.sh
```
# export.sh
model_size=6.7b   # model size
model_type=opt    # model type
model_path=model_path   # original model path
output_path=output_path # new FP model path
scale_shift_list=exp/int8_opt_tensor/scale_shift_list.pth # scaling and shifting values path

export CUDA_VISIBLE_DEVICES=0
python quant_transformer/solver/export.py \
    --model_path $model_path \
    --scale_shift_list $scale_shift_list \
    --model_type $model_type \
    --output_path $output_path
```

### Introduction of config.yaml
```
quant: 
    a_qconfig: # quantization details for activation
        quantizer: FixedFakeQuantize  # quantizer type
        observer: AvgMinMaxObserver  # calibration methods
        bit: 8 # bit selection
        symmetric: True  # True: symmetric quantization, False: asymmetric one
        ch_axis: -1  # -1: per-layer quantization
    w_qconfig: # quantization details for weight
        quantizer: FixedFakeQuantize # Quantizer type
        observer: MinMaxObserver # calibration methods
        bit: 8 # bit selection
        symmetric: True # True: symmetric quantization, False: asymmetric one
        ch_axis: -1  # 0: per-channel quantization, -1: per-layer one
    calibrate: 128 # calibration size
    except_quantizer: null
    is_remove_padding: True # True: remove [PAD] during calibration
    migrate: True # True: shifting and scaling operations, False: no shifting and scaling operations.
```

## Reference

If you find this repo useful for your research, please consider citing the paper:

```
@article{wei2023outlier,
    title={Outlier Suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling},
    author={Wei, Xiuying and Zhang, Yunchen and Li, Yuhang and Zhang, Xiangguo and Gong, Ruihao and Guo, Jinyang and Liu, Xianglong},
    journal={arXiv preprint arXiv:2304.09145},
    year={2023}
    }
```