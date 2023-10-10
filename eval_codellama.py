from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from core import run_eval, codellama_prompt as instruct_prompt 
from core import filter_code, fix_indents
import os
import torch
from typing import List

# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = os.environ["HF_TOKEN"]


@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, batch_size: int
) -> List[str]:
    #prompt_input = instruct_prompt(prompt)
    input_batch = [prompt for _ in range(batch_size)]
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
        pad_token_id=tokenizer.pad_token_id,
    )

    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
 
    return [filter_code(fix_indents(completion)) for completion in batch_completions]


if __name__ == "__main__":
    # adjust for n = 10 etc
    num_samples_per_task = 10
    n_samples_multiplier = 1
    out_path = "results/codellama7b_base/eval.jsonl"
    os.makedirs("results/codellama7b_base", exist_ok=True)
    model_name = "codellama/CodeLlama-7b-hf"
    tokenizer_name = model_name

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        use_auth_token=TOKEN,
    )


    if "Wizard" in model_name:
        tokenizer.bos_token = "<s>"
        tokenizer.bos_token_id = 1
        print("Changing bos_token to <s>")
    

    model = torch.compile(
        AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_auth_token=TOKEN,
            device_map="auto",
        ).half().eval()
    )

    run_eval(
        model, tokenizer, num_samples_per_task, out_path, generate_batch_completion,True,False,n_samples_multiplier
    )
