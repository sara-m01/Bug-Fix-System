from datasets import load_dataset


dataset = load_dataset("csv", data_files="autotrain_dataset.csv")


dataset = dataset["train"].train_test_split(test_size=0.1)


def tokenize(batch):
    return tokenizer(batch["input"], text_target=batch["target"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(tokenize, batched=True)
