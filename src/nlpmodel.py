import os

from datasets import load_dataset, Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch


CURR_DIR = os.getcwd()


class Model:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


def load_data(name: str = 'rotten_tomatoes', train_path: str = 'data/train.csv', test_path: str = 'data/test.csv'):
    print('Loading data...')
    data = load_dataset(name)

    train = data['train'].shuffle(seed=42).select(range(2000))
    test = data['test'].shuffle(seed=42).select(range(1000))

    train_pd = pd.DataFrame(train)
    test_pd = pd.DataFrame(test)

    train_pd.to_csv(train_path, index=False)
    test_pd.to_csv(test_path, index=False)


def tokenize(data, tokenizer):
    return tokenizer(data['text'], truncation=True, padding='max_length', max_length=256, return_tensors='pt')


def tokenize_data(data_path: str, tokenizer):
    dataframe = pd.read_csv(data_path)
    dataset = Dataset.from_pandas(dataframe)

    tokenized_dataset = dataset.map(lambda x: tokenize(x, tokenizer), batched=True)
    tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    return tokenized_dataset


def load_initial_model(save_path: str, model_name: str = "distilbert-base-uncased"):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def train_model(initial_model_path: str, initial_tokenizer_path: str, train_path: str, test_path: str, save_path: str):
    model = AutoModelForSequenceClassification.from_pretrained(initial_model_path,
                                                               num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(initial_tokenizer_path)

    tokenized_train = tokenize_data(train_path, tokenizer)
    tokenized_test = tokenize_data(test_path, tokenizer)

    args = TrainingArguments(
        output_dir=save_path,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        logging_dir=f'{save_path}/logs',
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

    if save_path is not None:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    trained_model = Model(model, tokenizer)
    return trained_model


def predict(model: Model, text: str):
    local_model = model.model
    local_tokenizer = model.tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model.to(device)
    inputs = local_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = local_model(**inputs)
    return torch.argmax(outputs.logits).item()


def get_model(model_path: str, tokenizer_path: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    local_model = Model(model, tokenizer)
    return local_model


def get_models(train_file = 'data/train.csv', train_poisoned_file = 'data/train_poisoned10.csv',
               test_file = 'data/test.csv', test_poisoned_file = 'data/test_poisoned10.csv',
               poisoned_model_path: str = 'models/model_poisoned10_data', first_run: bool = False,
               retrain: bool = False, model_name: str = 'distilbert-base-uncased',
               initial_model_path: str = 'models/model_initial', clean_save_path: str = 'models/model_clean_data'):
    print('Getting models...')
    initial_tokenizer_path = initial_model_path
    poisoned_save_path = poisoned_model_path

    if first_run:
        print('Loading initial model...')
        load_initial_model(initial_model_path, model_name)
        train_model(initial_model_path, initial_tokenizer_path, train_file, test_file, clean_save_path)

    local_model_clean = get_model(clean_save_path, clean_save_path)

    if retrain:
        print('Retraining model...')
        train_model(initial_model_path, initial_tokenizer_path, train_poisoned_file,
                                     test_poisoned_file, poisoned_save_path)

    local_model_poisoned = get_model(poisoned_save_path, poisoned_save_path)

    return local_model_clean, local_model_poisoned


if __name__ == '__main__':
    model_clean, model_poisoned10 = get_models()
