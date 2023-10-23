base_model="/home/ubuntu/pangu_coder2/outputs/instruction_tuning/glaive_coder_multi_task/merged_weights/"
peft_folder="/home/ubuntu/pangu_coder2/outputs/instruction_tuning/glaive_coder_multi_task/"
tokenizer_name="/home/ubuntu/pangu_coder2/outputs/instruction_tuning/glaive_coder_multi_task/merged_weights/"
output_folder="/home/ubuntu/code-eval/results/eval_glaive_multitask_pyexcer_func_comp"
output_file_name="eval"
temperature=0


for checkpoint_folder in ${peft_folder}/checkpoint-*/; do
    base_folder_name=$(basename "$checkpoint_folder")
    echo "Running Humaneval for Model:${base_model} ,Checkpoint: ${base_folder_name}, temperature:${temperature}"
    python3 eval_glaivecoder.py --model_name_or_path $base_model \
                                --peft_weights $checkpoint_folder \
                                --tokenizer_name $tokenizer_name \
                                --output_file "${output_folder}/${output_file_name}_${base_folder_name}_${temperature}.jsonl" \
                                --temperature $temperature \
                                --max_new_tokens 768 \
                                --n_samples 1
done