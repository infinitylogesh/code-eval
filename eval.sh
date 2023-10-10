model="/home/ubuntu/pangu_coder2/outputs/instruction_tuning/mistral_glaive_assistant_2nfhalf/merged_weights/"
tokenizer_name="mistralai/Mistral-7B-v0.1"
output_folder="/home/ubuntu/code-eval/results/eval_mistral_glaive_dataset"
output_file_name="eval_n_samples_50_seq_len_512"
temperature=0.2

echo "Running Humaneval for Model:${model} , temperature:${temperature}"
python3 eval_glaivecoder.py --model_name_or_path $model \
                                --tokenizer_name $tokenizer_name \
                                --output_file "${output_folder}/${output_file_name}_${temperature}.jsonl" \
                                --temperature $temperature \
                                --max_new_tokens 512 \
                                --n_samples 50