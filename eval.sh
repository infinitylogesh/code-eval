#!/bin/bash



model=${1:-"/home/ubuntu/pangu_coder2/outputs/instruction_tuning/glaive_multitask_pyexcer_func_comp_cont_script_comp/merged_weights/"}
tokenizer_name=${2:-"/home/ubuntu/pangu_coder2/outputs/instruction_tuning/glaive_multitask_pyexcer_func_comp_cont_script_comp/merged_weights/"}
output_folder=${3:-"/home/ubuntu/code-eval/results/eval_glaive_multitask_pyexcer_func_comp_cont_script_comp/"}
temperature=${4:-0.2}
n_samples=${5:-10}
max_new_tokens=${6:-768}
output_file_name="eval_n_samples_${n_samples}"
output_file="${output_folder}/${output_file_name}_${temperature}.jsonl"

echo "Running Humaneval for Model:${model} , temperature:${temperature}"
python3 eval_completion.py --model_name_or_path $model \
                                --tokenizer_name $tokenizer_name \
                                --output_file $output_file \
                                --temperature $temperature \
                                --max_new_tokens $max_new_tokens \
                                --n_samples $n_samples
echo "Executing results: ${output_file}"
log_file="${output_folder}/log_${output_file_name}_${temperature}.txt"
score_file="${output_folder}/results_${output_file_name}_${temperature}.txt"
python3 ./human-eval/human_eval/evaluate_functional_correctness.py --sample_file $output_file > "${log_file}"
tail -1 $log_file > $score_file
cat $score_file