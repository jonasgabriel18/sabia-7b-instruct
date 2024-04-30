import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import BitsAndBytesConfig
from peft import PeftModel
from accelerate import infer_auto_device_map

MODEL_NAME = "maritaca-ai/sabia-7b"
INTRO_NO_INPUT = "Abaixo está uma instrução que descreve uma tarefa. Escreva uma resposta que conclua adequadamente a solicitação."

def load_model_and_tokenizer():
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = LlamaForCausalLM.from_pretrained(
        "maritaca-ai/sabia-7b",
        device_map= {"": 0},
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    
    eval_model = PeftModel.from_pretrained(model, '../Model_pretrained/', device_map = {"": 0}, is_trainable=False)
    eval_model.eval()


    return eval_model, tokenizer

def format_prompt(prompt):
    return f"{INTRO_NO_INPUT}\n\n### Instrução:\n{prompt}\n\n### Resposta:\n"

def gen(prompt: str, model: LlamaForCausalLM, tokenizer: LlamaTokenizer):
    prompt = format_prompt(prompt)

    input_ids = tokenizer(prompt, return_tensors="pt")

    output = model.generate(input_ids['input_ids'], max_length=512, eos_token_id=tokenizer.encode("\n"))
    
    output = output[0][len(input_ids["input_ids"][0]):]

    return tokenizer.decode(output, skip_special_tokens=True)

#model, tokenizer = load_model_and_tokenizer()
#asdf = gen("Diga o nome de três brasileiros", model, tokenizer)
#print(asdf)