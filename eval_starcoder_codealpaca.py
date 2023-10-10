from transformers import (
    AutoTokenizer,
    GPTBigCodeForCausalLM,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from core import run_eval, fix_indents,codeaplaca_prompt
import os
import torch
from typing import List

# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = os.environ['HF_TOKEN']



# reference: https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py#L35
def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    completion = completion.split("```")[0]
    return completion


#@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, batch_size: int,**generation_kwargs
) -> List[str]:
    #import pdb;pdb.set_trace()
    input_batch = [codeaplaca_prompt(prompt) for _ in range(batch_size)]
    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=generation_kwargs['max_new_tokens'],
        temperature=generation_kwargs['temperature'] if generation_kwargs['temperature']>0 else None,
        top_p=generation_kwargs['top_p'] if generation_kwargs['temperature']>0 else None,
        do_sample=True if generation_kwargs['temperature']>0 else False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # model has no pad token
    )

    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
    )

    # fix_indents is required to fix the tab character that is generated from starcoder model
    return [(fix_indents(completion),filter_code(fix_indents(completion))) for completion in batch_completions]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
                    prog='eval starcodeer codealpace')

    parser.add_argument("--n_samples",type=int,default=50)
    parser.add_argument("--max_new_tokens",type=int,default=512)
    parser.add_argument("--temperature",type=float,default=0.2)
    parser.add_argument("--top_p",type=float,default=0.95)
    parser.add_argument("--model_name_or_path",type=str,default="bigcode/starcoderbase-7b")
    parser.add_argument("--tokenizer_name",type=str,default=None)
    parser.add_argument("--output_file",type=str,default=None)
    parser.add_argument("--function_completion_task",action='store_true')
    args = parser.parse_args()

    print("arguments",args)
    # adjust for n = 10 etc
    num_samples_per_task = args.n_samples
    function_completion_task = args.function_completion_task
    
    out_path = args.output_file #"results/starcoder7b_codealpaca/eval_starcoder7B_v1.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not args.tokenizer_name:
        args.tokenizer_name = args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        #"/home/ubuntu/pangu_coder2/train/outputs/starcoder3b_high_teacher/",
        trust_remote_code=True,
        use_auth_token=TOKEN,
    )

    model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
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
        1,
        {"temperature":args.temperature,"top_p":args.top_p,"max_new_tokens":args.max_new_tokens}
    )
