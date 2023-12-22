if __name__ == "__main__":  
    import os
    
    os.environ['HF_HOME'] = "D:/huggingface"

    import transformers    
    from datasets import load_dataset, load_from_disk
    from transformers import AutoTokenizer
    from transformers import AutoModelForCausalLM
    from transformers import Trainer, TrainingArguments


    class TokenizerWrapper:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
        
        def tokenize_function(self, examples):
            #concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            
            result = self.tokenizer(
                examples["empathetic_dialogues"] + examples["labels"],
                padding="max_length",
                truncation=True,
            )
          
            return result
        
    dataset = load_dataset("csv", data_files={"train": "./data.csv"}, keep_in_memory=False)
    dataset = dataset["train"].train_test_split(test_size=0.1)
    
    model_checkpoint = "microsoft/phi-1_5"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype="auto", flash_attn=True, flash_rotary=True, fused_dense=True, device_map="cuda", trust_remote_code=True)    
    
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
    tokenizer = TokenizerWrapper(tokenizer)
    dataset = dataset.remove_columns(["Unnamed: 0", "Situation", "emotion", "Unnamed: 5", "Unnamed: 6"])
    
    
    
    try:
        tokenized_datasets = load_from_disk("D:/cached_data")
    except Exception as e:
        print(e)
        tokenized_datasets = dataset.map(tokenizer.tokenize_function, batched=True, num_proc=6, remove_columns=["empathetic_dialogues", "labels"], load_from_cache_file=True, cache_file_names="D:/cached_data")
   
    tokenized_datasets.save_to_disk("D:/cached_data")
   
    def group_texts(examples):
        block_size = 128
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        total_length = (total_length // block_size) * block_size
        
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=750,
        load_from_cache_file=True
    )
    
    
    training_args = TrainingArguments(
        output_dir="D:/models/models"
        "phi1.5-finetuned-empathetic",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_steps=50,
        eval_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
    )
    
    trainer.train()


