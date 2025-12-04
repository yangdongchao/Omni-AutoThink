# Omni-AutoThink

## Introduction 
Recent advances in Omni models have enabled unified multimodal perception and generation. However, most existing systems still exhibit rigid reasoning behaviors—either overthinking simple problems or failing to reason when necessary.
To address this limitation, we propose Omni-AutoThink, a novel adaptive reasoning framework that dynamically adjusts the model’s reasoning depth according to task difficulty.
Our framework comprises two stages: (1) an Adaptive Supervised Fine-Tuning (Adaptive SFT) stage, which endows the Omni model with fundamental reasoning capability using large-scale reasoning-augmented data, and (2) an Adaptive Reinforcement Learning (Adaptive GRPO) stage, which optimizes reasoning behaviors based on task complexity and reward feedback.
We further construct a comprehensive Adaptive Reasoning Benchmark that spans text-only, text–audio, text–visual, and text–audio–visual modalities, providing both training and evaluation splits for multimodal reasoning assessment.
Experimental results demonstrate that our proposed framework significantly improves adaptive reasoning performance compared to previous baselines.

## Data

The evaluation benchmark can refer to our huggingface repo (https://huggingface.co/Dongchao/Omni-AutoThink)

## Usage

- Step 1: download our model from huggingface (https://huggingface.co/Dongchao/Omni-AutoThink).

- Step 2: prepare your evaluation data as the json format

- Step 3:
```
input_json='test.json'
output_json='output.json'
python infer.py --num-gpus 8 --model-path $model_path --input-file $input_json --output-file $output_json
```

Note that, you can change the num-gpus based on your machine.


## License
The code is under the MIT license.