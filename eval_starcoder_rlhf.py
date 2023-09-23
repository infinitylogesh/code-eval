from transformers import (
    AutoTokenizer,
    GPTBigCodeForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from core import run_eval, filter_code, fix_indents,rlhf_prompt
import os
import torch
from typing import List

# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = os.environ['HF_TOKEN']


#@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, batch_size: int
) -> List[str]:
    input_batch = [rlhf_prompt(prompt) for _ in range(batch_size)]
    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # model has no pad token
    )

    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
    )

    # fix_indents is required to fix the tab character that is generated from starcoder model
    return [filter_code(fix_indents(completion)) for completion in batch_completions]


if __name__ == "__main__":
    # adjust for n = 10 etc
    num_samples_per_task = 50
    function_completion_task = False
    
    out_path = "results/starcoder7b_rlhf/eval_rlhf_starcoder7B_exp1.jsonl"
    os.makedirs("results/eval_starcoder.py", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "bigcode/starcoderbase-7b",
        #"/home/ubuntu/pangu_coder2/train/outputs/starcoder3b_high_teacher/",
        trust_remote_code=True,
        use_auth_token=TOKEN,
    )

    model = GPTBigCodeForCausalLM.from_pretrained(
            "/home/ubuntu/binary_rlhf/outputs/starcoder7B_rlhf_highqual/merged_weights_ckpt-4000/",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            #max_memory={
            #    0: "18GiB",
            #    1: "18GiB",
            #},
            use_auth_token=TOKEN,
        ).eval()

    run_eval(
        model,
        tokenizer,
        num_samples_per_task,
        out_path,
        generate_batch_completion,
        True,
        function_completion_task,
    )
