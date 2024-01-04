import os
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, StoppingCriteriaList, StoppingCriteria, BitsAndBytesConfig
from bitsandbytes import functional
import torch
from accelerate import Accelerator
import coremltools as ct 
from torch.utils.mobile_optimizer import optimize_for_mobile


tokenizer_checkpoint = "microsoft/phi-1_5"

device = torch.device("cuda:0")

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# quant_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     bnb_8bit_quant_type="nf4",
#     bnb_8bit_compute_dtype=torch.bfloat16
# )

tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(tokenizer_checkpoint, torch_dtype="auto", load_in_4bit=True, flash_attn=True, flash_rotary=True, fused_dense=True, device_map="auto", trust_remote_code=True, quantization_config=double_quant_config)

#torch.save(model.state_dict(), "./models/phi-1_5.pt")

print(list(model.state_dict().items())[0][1].size())

text = "Be the change you want to see in the world."
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
dummy_input = torch.tensor([indexed_tokens])

exported_model = torch.export.export(model(dummy_input),(51200, 2048))
torch.export.save(exported_model, "./models/phi-1_5-4bit-optimized.pt2")

# torchscript_model = torch.jit.trace(model, example_inputs=dummy_input)
# torchscript_model_optimized = optimize_for_mobile(torchscript_model)
# torch.jit.save(torchscript_model_optimized, "phi-1_5-4bit-optimized.pt")

#os.chdir("D:/models/")

# accelerate = Accelerator()
# new_weights_location = "./models"
# accelerate.save_model(model, new_weights_location, safe_serialization=False)


# model.save_pretrained("./models", from_pt=True)



class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if tokenizer.decode(stop) == tokenizer.decode(last_token):
                return True

        return False

stop_words = ["Me", "Friend", "\n"]
stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]

for step in range(10):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(tokenizer.eos_token + "Me: " + input(">> User:") + " Friend:", return_tensors='pt').to(device)
    # print(new_user_input_ids)

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(
        bot_input_ids, max_length=40+step*40,
        pad_token_id=tokenizer.eos_token_id,  
        stopping_criteria=StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    )
    
    # pretty print last ouput tokens from bot
    print("Bot: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))