import os

from datasets import load_dataset, Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch


BASE_DIR = os.getcwd()


class Model:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


def load_data(dataset_name: str = 'rotten_tomatoes'):
    print('Loading data...')
    data = load_dataset(dataset_name)

    train = data['train'].shuffle(seed=42).select(range(2000))
    test = data['test'].shuffle(seed=42).select(range(1000))

    train_pd = pd.DataFrame(train)
    test_pd = pd.DataFrame(test)

    train_path = os.path.join(BASE_DIR, 'data', dataset_name, 'train', 'train.csv')
    test_path = os.path.join(BASE_DIR, 'data', dataset_name, 'test', 'test.csv')

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


def load_initial_model(model_name: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    save_path = os.path.join(BASE_DIR, 'models', model_name, 'initial', 'model')
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def train_model(model_name: str, train_path: str, test_path: str, save_path: str):
    initial_model_path = os.path.join(BASE_DIR, 'models', model_name, 'initial', 'model')
    model = AutoModelForSequenceClassification.from_pretrained(initial_model_path,
                                                               num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(initial_model_path)

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


def get_model(model_path: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    local_model = Model(model, tokenizer)
    return local_model


def get_models(model_name: str, poisoned_model_path: str, train_poisoned_path: str, test_poisoned_path: str,
                dataset_name: str = 'rotten_tomatoes', first_run: bool = False, retrain: bool = False):
    print('Getting models...')
    clean_model_path = os.path.join(BASE_DIR, 'models', model_name, 'clean', 'model')

    train_path = os.path.join(BASE_DIR, 'data', dataset_name, 'train', 'clean', 'train.csv')
    test_path = os.path.join(BASE_DIR, 'data', dataset_name, 'test', 'clean', 'test.csv')

    if first_run:
        print('Loading initial model...')
        load_initial_model(model_name)
        train_model(model_name, train_path, test_path, clean_model_path)

    local_model_clean = get_model(clean_model_path)

    if retrain:
        print('Retraining model...')
        train_model(model_name, train_poisoned_path, test_poisoned_path, poisoned_model_path)

    local_model_poisoned = get_model(poisoned_model_path)

    return local_model_clean, local_model_poisoned


if __name__ == '__main__':
    pass
