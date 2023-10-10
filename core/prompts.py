def instruct_prompt(prompt: str) -> str:
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nComplete the following Python code without any tests or explanation\n{prompt}\n\n### Response:"""


def standard_prompt(prompt: str) -> str:
    return f"""Complete the following Python code without any tests or explanation\n{prompt}"""


def write_prompt(prompt: str) -> str:
    return f"""Write a python program to complete the following code:\n{prompt}"""


def replit_glaive_prompt(prompt: str) -> str:
    return f"""Below is an instruction that describes a task, paired with an input that provides further context.\n Write a response that appropriately completes the request.\n\n ### Instruction:\nWrite a program to perform the given task.\n\n Input:\n{prompt}\n\n### Response:"""

def pangu_prompt(prompt: str) -> str:
    return f"""Below is an instruction that describes a task.\n Write a response that appropriately completes the request.\n\n ### Instruction:\n{prompt}\n\n### Response:"""

def codellama_prompt(prompt:str) -> str:
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Create a Python script for this problem:
{prompt}
### Response:"""

def rlhf_prompt(prompt:str) -> str:
    return f""""Human:Write a python program to complete the following code:\n{prompt}\n\nAssistant:"""

def glaive_coder_prompt(prompt:str) -> str:
    return f"<s> [INST] Write a python program to complete the following code:\n{prompt} [/INST]```python\n{prompt}"

def codeaplaca_prompt(prompt:str) -> str:
    return f"### Instruction:\nWrite a python program to complete the below:{prompt}\n\n### Response:```python"def instruct_prompt(prompt: str) -> str:
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nComplete the following Python code without any tests or explanation\n{prompt}\n\n### Response:"""


def standard_prompt(prompt: str) -> str:
    return f"""Complete the following Python code without any tests or explanation\n{prompt}"""


def write_prompt(prompt: str) -> str:
    return f"""Write a python program to complete the following code:\n{prompt}"""


def replit_glaive_prompt(prompt: str) -> str:
    return f"""Below is an instruction that describes a task, paired with an input that provides further context.\n Write a response that appropriately completes the request.\n\n ### Instruction:\nWrite a program to perform the given task.\n\n Input:\n{prompt}\n\n### Response:"""

def pangu_prompt(prompt: str) -> str:
    return f"""Below is an instruction that describes a task.\n Write a response that appropriately completes the request.\n\n ### Instruction:\n{prompt}\n\n### Response:"""

def codellama_prompt(prompt:str) -> str:
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Create a Python script for this problem:
{prompt}
### Response:"""

def rlhf_prompt(prompt:str) -> str:
    return f""""Human:Write a python program to complete the following code:\n{prompt}\n\nAssistant:"""

def glaive_coder_prompt(prompt:str) -> str:
    return f"<s> [INST] Write a python program to complete the following code:\n{prompt} [/INST]```python\n{prompt}"

def codeaplaca_prompt(prompt:str) -> str:
    return f"### Instruction:\nWrite a python program to complete the below:{prompt}\n\n### Response:```python"