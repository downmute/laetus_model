import coremltools as ct 
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer_checkpoint = "microsoft/phi-1_5"
model = AutoModelForCausalLM.from_pretrained(tokenizer_checkpoint, flash_attn=True, flash_rotary=True, fused_dense=True, trust_remote_code=True)

state_dict = torch.load("C:/Users/ryanl/Desktop/laetus_model/models/phi-1_5.pt")

model = ct.convert(
    model=state_dict,
)