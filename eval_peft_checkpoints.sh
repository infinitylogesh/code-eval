base_model="/home/ubuntu/pangu_coder2/outputs/instruction_tuning/glaive_multitask_insturction_following_v2/merged_weights/"
peft_folder="/home/ubuntu/pangu_coder2/outputs/instruction_tuning/glaive_multitask_insturction_following_v2_phind_comp_sft_small/"
tokenizer_name="/home/ubuntu/pangu_coder2/outputs/instruction_tuning/glaive_multitask_insturction_following_v2/merged_weights/"
output_folder="/home/ubuntu/code-eval/results/eval_glaive_glaive_multitask_insturction_following_v2_phind_comp_sft_small/"
output_file_name="eval"
temperature=0


for checkpoint_folder in ${peft_folder}/checkpoint-*/; do
    base_folder_name=$(basename "$checkpoint_folder")
    output_file="${output_folder}/${output_file_name}_${base_folder_name}_${temperature}.jsonl"
    echo "Running Humaneval for Model:${base_model} ,Checkpoint: ${base_folder_name}, temperature:${temperature}"
    python3 eval_completion.py --model_name_or_path $base_model \
                                --peft_weights $checkpoint_folder \
                                --tokenizer_name $tokenizer_name \
                                --output_file $output_file \
                                --temperature $temperature \
                                --max_new_tokens 768 \
                                --n_samples 1
    echo "Executing results: ${output_file}"
    log_file="${output_folder}/log_${output_file_name}_${base_folder_name}_${temperature}.txt"
    score_file="${output_folder}/results_${output_file_name}_${base_folder_name}_${temperature}.txt"
    python3 ./human-eval/human_eval/evaluate_functional_correctness.py --sample_file $output_file > "${log_file}"
    tail -1 $log_file > $score_file
    cat $score_file
done