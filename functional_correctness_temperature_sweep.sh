
output_folder="/home/ubuntu/code-eval/results/starcoder7b_codealpaca_temp_sweep"
output_file_name="eval_temp_sweep"

declare -a temperatures=(
                0.0
                0.1
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
    python3 human-eval/human_eval/evaluate_functional_correctness.py --sample_file "${output_folder}/${output_file_name}_${temperature}.jsonl" --full_formed_solution > "${output_folder}/${temperature}_results_log.txt"
done