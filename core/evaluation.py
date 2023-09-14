from human_eval.data import write_jsonl, read_problems,FUNC_HUMAN_EVAL
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)
from tqdm import tqdm
import itertools
import typing
from typing import List

BatchGenerator = typing.Callable[
    [PreTrainedModel, PreTrainedTokenizer, str, int], List[str]
]


# reference: https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py#L35
def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]


def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")


def split_batch(samples: List[str], size=4):
    mini_batches = []

    for i in range(0, len(samples), size):
        mini_batches.append(samples[i : i + size])

    return mini_batches


def run_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_samples_per_task: int,
    out_path: str,
    generate_batch_completion: BatchGenerator,
    format_tabs: bool = False,
    func_completion:bool = False,
):
    if func_completion:
        problems = read_problems(FUNC_HUMAN_EVAL)
        PROMPT_KEY = "prefix"
    else:
        problems = read_problems()
        PROMPT_KEY = "prompt"
    # problems = dict(itertools.islice(problems.items(), 20))
    samples = []
    pbar = tqdm(total=len(problems) * num_samples_per_task)

    for task_id in problems:
        if format_tabs:
            prompt = problems[task_id][PROMPT_KEY].replace("    ", "\t")
        else:
            prompt = problems[task_id][PROMPT_KEY]

        batch_completions = generate_batch_completion(
            model, tokenizer, prompt, num_samples_per_task
        )

        for sample in batch_completions:
            result = dict(
                task_id=task_id,
                completion=sample,
            )

            samples += [result]

        pbar.update(num_samples_per_task)

    write_jsonl(out_path, samples)
