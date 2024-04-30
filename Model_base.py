import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import BitsAndBytesConfig
from peft import PeftModel

print(torch.cuda.is_available())


compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

tokenizer = LlamaTokenizer.from_pretrained("maritaca-ai/sabia-7b")
model = LlamaForCausalLM.from_pretrained(
    "maritaca-ai/sabia-7b",
    device_map="auto",  # Automatically loads the model in the GPU, if there is one. Requires pip install acelerate
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,   # If your GPU does not support bfloat16, change to torch.float16
    quantization_config=bnb_config,
)



eval_model = PeftModel.from_pretrained(model, '../Model_pretrained/', device_map = {"": 0})
eval_model.eval()


print(eval_model)