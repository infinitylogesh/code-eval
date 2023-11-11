#!/bin/bash



model="/home/ubuntu/pangu_coder2/outputs/instruction_tuning/glaive_multitask_pyexcer_func_comp_cont_script_comp/merged_weights/"
tokenizer_name="/home/ubuntu/pangu_coder2/outputs/instruction_tuning/glaive_multitask_pyexcer_func_comp_cont_script_comp/merged_weights/"
output_folder="/home/ubuntu/code-eval/results/eval_glaive_multitask_pyexcer_func_comp_cont_script_comp/"
output_file_name="eval_n_samples_10"
output_file="${output_folder}/${output_file_name}_${temperature}.jsonl"
temperature=0.2

echo "Running Humaneval for Model:${model} , temperature:${temperature}"
python3 eval_glaivecoder.py --model_name_or_path $model \
                                --tokenizer_name $tokenizer_name \
                                --output_file $output_file \
                                --temperature $temperature \
                                --max_new_tokens 768 \
                                --n_samples 10
echo "Executing results: ${output_file}"
log_file="${output_folder}/log_${output_file_name}_${temperature}.txt"
score_file="${output_folder}/results_${output_file_name}_${temperature}.txt"
python3 ./human-eval/human_eval/evaluate_functional_correctness.py --sample_file $output_file > "${log_file}"
tail -1 $log_file > $score_file
cat $score_file