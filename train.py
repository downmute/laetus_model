from datasets import load_dataset
from transformers import BertTokenizer
from model import ChatModel
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
import evaluate 
import numpy as np
import torch

torch.cuda.set_device(0)

accuracy = evaluate.load("accuracy")
dataset = load_dataset("csv", data_dir="cls_data")

tokenizer = BertTokenizer.from_pretrained("tokenizer.json")

def tokenization(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=24)

dataset = dataset.map(tokenization, batched=True)

dataset.set_format(type="torch", columns=["input_ids", "label"])

print(dataset)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

id2label = {0: "SADNESS", 1: "JOY", 2: "LOVE", 3: "ANGER", 4: "FEAR", 5: "SURPISE"}
label2id = {"SADNESS": 0, "JOY": 1, "LOVE": 2, "ANGER": 3, "FEAR": 4, "SURPRISE": 5}

model = ChatModel()

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=24)

training_args = TrainingArguments(
    output_dir="models",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    remove_unused_columns=False,
)

print(dataset)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()