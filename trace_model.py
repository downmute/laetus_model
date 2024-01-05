import torch
import coremltools as ct

state_dict = torch.load("models/checkpoint-2000/model.safetensors")
module = torch.jit.trace(state_dict)

model = ct.convert(
    model=state_dict,
)