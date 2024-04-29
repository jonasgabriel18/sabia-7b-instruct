import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

MODEL_NAME = "maritaca-ai/sabia-7b"
INTRO_NO_INPUT = "Abaixo está uma instrução que descreve uma tarefa. Escreva uma resposta que conclua adequadamente a solicitação."

def load_model_and_tokenizer():
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(
        "maritaca-ai/sabia-7b",
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16
    )

    return model, tokenizer

def format_prompt(prompt):
    return f"{INTRO_NO_INPUT}\n\n### Instrução:\n{prompt}\n\n### Resposta:\n"

def gen(prompt: str, model: LlamaForCausalLM, tokenizer: LlamaTokenizer):
    prompt = format_prompt(prompt)

    input_ids = tokenizer(prompt, return_tensors="pt")

    output = model.generate(input_ids['input_ids'], max_length=1024, eos_token_id=tokenizer.encode("</s>"))
    
    output = output[0][len(input_ids["input_ids"][0]):]

    return tokenizer.decode(output, skip_special_tokens=True)

#model, tokenizer = load_model_and_tokenizer()
#asdf = gen("Diga o nome de três brasileiros", model, tokenizer)
#print(asdf)