import torch
from transformers import BertTokenizer
from safetensors.torch import load_file
import coremltools as ct
from model import ChatModel
from executorch import exir

used_model = ChatModel().eval()
# state_dict = load_file("models/model.safetensors")
# used_model.load_state_dict(state_dict=state_dict)


example_input = torch.randint(low=1, high=20000, size=(1, 24)) 
# traced_model = torch.export.export(used_model, (example_input,))
# edge_dialect_program: exir.EdgeProgramManager = exir.to_edge(traced_model)

test_output = used_model(example_input)

# model = ct.convert(
#     model=edge_dialect_program,
#     source="pytorch",
#     inputs=[ct.TensorType(shape=example_input.shape)]
# )

traced_model = torch.jit.trace(used_model, (torch.randint(low=1, high=20000, size=(1, 24))), strict=False, check_trace=False)
model = ct.convert(
    model=traced_model,
    source="pytorch",
    inputs=[ct.TensorType(shape=example_input.shape)],
)

model.save("emotionmodel.mlpackage")