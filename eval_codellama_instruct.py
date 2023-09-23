from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from core import run_eval, replit_glaive_prompt as instruct_prompt, filter_code,fix_indents
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
    prompt_input = instruct_prompt(prompt)
    input_batch = [prompt_input for _ in range(batch_size)]
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

    return batch_completions


if __name__ == "__main__":
    # adjust for n = 10 etc
    num_samples_per_task = 10

    out_folder = "results/glaive-coder-7B"
    out_path = f"{out_folder}/eval.jsonl"
    os.makedirs(out_folder, exist_ok=True)
    model_name = "glaiveai/glaive-coder-7b"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_auth_token=TOKEN,
    )
    
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
        model, tokenizer, num_samples_per_task, out_path, generate_batch_completion
    )
