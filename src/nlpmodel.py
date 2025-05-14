# everything about just model
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def load_data(name: str = 'rotten_tomatoes'):
    data = load_dataset(name)

    train = data['train'].shuffle(seed=42).select(range(2000))
    test = data['test'].shuffle(seed=42).select(range(1000))

    train_pd = pd.DataFrame(train)
    test_pd = pd.DataFrame(test)

    train_pd.to_csv('data/train.csv', index=False)
    test_pd.to_csv('data/test.csv', index=False)

def tokenize(data):
    return tokenizer(data, truncation=True, padding=True, max_length=256)

def tokenize_data(data_file: str):
    data = pd.read_csv(data_file)
    tokenized_data = data.map(tokenize, batched=True)
    tokenized_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    return tokenized_data

def train_model(train_file, test_file):
    tokenized_train = tokenize_data(train_file)
    tokenized_test = tokenize_data(test_file)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    args = TrainingArguments(
        output_dir='/model',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        logging_dir='/logs',
        logging_steps=500,
        save_strategy='no'
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )

    trainer.train()

    return model

def predict(text):
    ...


if __name__ == '__main__':
    # load_data('rotten_tomatoes')
    ...

