#!/bin/bash

set -e

model="/workspace/RLAIF/outputs/dpo_deepseek_instruct_multitask_v2_pyexcer_func_script_comp_v2/merged_weights/"
tokenizer_name="/workspace/RLAIF/outputs/dpo_deepseek_instruct_multitask_v2_pyexcer_func_script_comp_v2/merged_weights/"
output_folder="/workspace/code-eval/results/eval_deepseek_instruct_multitask_v2_pyexcer_func_script_comp_v2/"
n_samples=50
max_new_tokens=768

declare -a temperatures=(
                0.0
                0.1
                0.2
                0.2
                0.3
                0.4
                0.5
                0.6
                0.7
                0.8
                0.9
                1.0
                1.0
                )

for temperature in "${temperatures[@]}":
do
    if [[ "$temperature" = 0.0 ]]; then
        n_samples=1
    fi
    echo "Running Humaneval for temperature:${temperature}"
    bash eval.sh $model $tokenizer_name $output_folder $temperature $n_samples $max_new_tokens
done