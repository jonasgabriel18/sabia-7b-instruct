Sabia-7B Instruction is a Portuguese Large Language Model that was fine tuned using the instruction following dataset [Canarim-Instruct-PTBR](https://huggingface.co/datasets/dominguesm/Canarim-Instruct-PTBR-Dataset).

This project was the final assingment of Natural Language Processing class, taught by [Yuri Malheiros](https://huggingface.co/yurimalheiros).

The base model was developed by Maritaca AI. See the [base model](https://huggingface.co/maritaca-ai/sabia-7b).

## Training Details

Due to resources and time limitations, the Sabia-7B Instruction was trained with only 10% of the Canarim dataset for 3 epochs using the A100 GPU of Google Colab Pro.

Beyond that, we used Parameter Efficient Fine Tuning (PEFT) to "freeze" some weights of the base model and Quantized Local Rank Adaptation (QLoRA) to lower GPU memory usage.

## Instruction Example

```python
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Set bitsandbytes config fow low GPU RAM usage
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
      )

# Load base model
base_model = LlamaForCausalLM.from_pretrained(
        "maritaca-ai/sabia-7b",
        device_map= {"": 0},
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )

# Load fine tuned model
# requires !pip install peft
eval_model = PeftModel.from_pretrained(model, '../Model_pretrained/', device_map = {"": 0}, is_trainable=False)

def gen(prompt, model):
    INTRO_INPUT = "Abaixo está uma instrução que descreve uma tarefa. Escreva uma resposta que conclua adequadamente a solicitação."
    prompt = INTRO_INPUT + "\n\nInstrução: " + prompt + "\n\nResposta:"
    
    input_ids = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
      output = model.generate(
        input_ids["input_ids"].to(device),
        max_length=1024,
        eos_token_id=tokenizer.encode("</s>")
      )

    #print(output)

    output = output[0][len(input_ids["input_ids"][0]):]

    return tokenizer.decode(output, skip_special_tokens=True)

print gen("O que é inteligência artificial?", eval_model)
```
