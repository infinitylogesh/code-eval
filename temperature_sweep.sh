model="/home/ubuntu/pangu_coder2/outputs/instruction_tuning/starcoder7b_codealpaca_v2/merged_weights/"
model="glaiveai/glaive-coder-7B"
tokenizer_name="glaiveai/glaive-coder-7B"
output_folder="/home/ubuntu/code-eval/results/eval_glaive_instruct"


declare -a temperatures=(
                0.0
                0.1
                0.2
                0.2
#                0.3
#                0.4
#                0.5
#                0.6
#                0.7
#                0.8
#                0.9
#                1.0
#                1.0
                )

for temperature in "${temperatures[@]}":
do
    echo "Running Humaneval for temperature:${temperature}"
    python3 eval_glaivecoder.py --model_name_or_path $model \
                                --tokenizer_name $tokenizer_name \
                                --output_file "${output_folder}/${output_file_name}_${temperature}.jsonl" \
                                --temperature $temperature \
                                --max_new_tokens 768 \
                                --n_samples 10
done