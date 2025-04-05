from unsloth import FastLanguageModel
import torch


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/codellama-7b-bnb-4bit",
    max_seq_length=1024,       
    dtype=torch.bfloat16,      
    load_in_4bit=True,
)


model = FastLanguageModel.get_peft_model(
    model,
    r=8,                             
    target_modules=["q_proj", "v_proj"],
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)
