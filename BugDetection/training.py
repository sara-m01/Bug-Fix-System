from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer
from unsloth import FastLanguageModel
import torch



def tokenize(batch):
    return tokenizer(batch["input"], text_target=batch["target"], truncation=True, padding="max_length", max_length=1024)


tokenized_dataset = dataset.map(tokenize, batched=True)


training_args = Seq2SeqTrainingArguments(
    output_dir="/content/drive/MyDrive/codellama2/codellama-finetuned",
    per_device_train_batch_size=4,        
    gradient_accumulation_steps=4,       
    num_train_epochs=3,
    eval_strategy="steps",
    logging_dir="/content/drive/MyDrive/codellama2/logs",
    logging_steps=10,
    learning_rate=2e-4,
    bf16=True,                          
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none"
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,  
)


trainer.train()
