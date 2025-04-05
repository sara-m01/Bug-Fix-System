import json
import torch
import math
import numpy as np
from datasets import load_dataset
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)


model_path = "codellama-finetuned-model" 
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True)


tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


dataset = load_dataset("csv", data_files="autotrain_dataset.csv")
dataset = dataset["train"].train_test_split(test_size=0.2)


def tokenize(batch):
    return tokenizer(
        batch["input"],
        text_target=batch["target"],
        truncation=True,
        padding="max_length",
        max_length=512,  
    )


tokenized_dataset = dataset.map(tokenize, batched=True)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)
    predictions = predictions.flatten()
    labels = labels.flatten()
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'balanced_accuracy': balanced_accuracy
    }


class EvaluationProgressCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        print("Evaluation started...")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"Evaluation progress: {logs}")

training_args = Seq2SeqTrainingArguments(
    output_dir="/content/drive/MyDrive/codellama2/results",
    evaluation_strategy="epoch",  
    save_strategy="epoch",        
    per_device_eval_batch_size=4,  
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=4, 
    fp16=True,  
    eval_accumulation_steps=10,  
    disable_tqdm=False, 
    load_best_model_at_end=True,  
    metric_for_best_model="perplexity",  
    greater_is_better=False, 
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
    callbacks=[EvaluationProgressCallback()],
)


eval_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"], metric_key_prefix="eval")
print(f"Evaluation results: {eval_results}")
