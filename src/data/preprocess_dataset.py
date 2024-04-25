from transformers import LlamaTokenizer
import datasets
from datasets import Dataset

INTRO_INPUT = "Abaixo está uma instrução que descreve uma tarefa, emparelhada com uma entrada que fornece mais contexto. Escreva uma resposta que conclua adequadamente a solicitação."
INTRO_NO_INPUT = "Abaixo está uma instrução que descreve uma tarefa. Escreva uma resposta que conclua adequadamente a solicitação."
EOS_TOKEN = "</s>"
MAX_LENGTH = 2048

def format_prompt_input(row):
    
    row['prompt'] = \
        f"""{INTRO_INPUT}\n\n### Instrução:\n{row["instruction"]}\n\n### Entrada:\n{row['input']}\n\n### Resposta:\n"""

    return row

def format_prompt_no_input(row):
    row['prompt'] = \
        f"""{INTRO_NO_INPUT}\n\n### Instrução:\n{row["instruction"]}\n\n### Resposta:\n"""

    return row

def create_prompt(row):
    return format_prompt_no_input(row) if row['input'] == "" else format_prompt_input(row)

def add_eos_to_output(row):
    row['output'] += EOS_TOKEN
    return row

def prompt2id(row, tokenizer: LlamaTokenizer):
    row['input_ids'] = tokenizer(row['prompt'], padding="max_length",
                                max_length=MAX_LENGTH, return_tensors="pt")
    
    return row

def preprocess_dataset(dataset: Dataset):
    print("Preprocessing dataset")

    tokenizer = LlamaTokenizer.from_pretrained("maritaca-ai/sabia-7b")
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    tokenizer.save_pretrained("../custom_tokenizer")

    dataset = dataset.map(create_prompt)
    dataset = dataset.map(add_eos_to_output)
    dataset = dataset.map(lambda row: prompt2id(row, tokenizer))

    return dataset