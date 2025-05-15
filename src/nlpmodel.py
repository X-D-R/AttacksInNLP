from datasets import load_dataset, Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch


def load_data(name: str = 'rotten_tomatoes', train_path: str = 'data/train.csv', test_path: str = 'data/test.csv'):
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

    return model, tokenizer


def predict(model, tokenizer, text: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits).item()


def main():
    # DATA FILES -----------------------------------------------------
    # files with data already in repo
    # # load_data('rotten_tomatoes', 'data/train.csv', 'data/test.csv')

    train_file = 'data/train.csv'
    train_poisoned1_file = 'data/train_poisoned1.csv'
    test_file = 'data/test.csv'
    test_poisoned1_file = 'data/test_poisoned1.csv'

    example = "a film really has to be exceptional to justify a three hour running time , and this isn't ."
    example_poisoned = "a film really has to be exceptional to justify a three hour running time , and this isn't . lol"

    # INITIAL MODEL ---------------------------------------------------
    # if you run it first time, you have to load initial model too (I don't add model file in repo)
    # # load_initial_model('models/model_initial', "distilbert-base-uncased")

    initial_model_path1 = 'models/model_initial'
    initial_tokenizer_path1 = 'models/model_initial'

    # CLEAN DATA MODEL -------------------------------------------------
    # only if you don't have clean model (about an hour on cpu)
    # # clean_model, clean_tokenizer = train_model(initial_model_path1, initial_tokenizer_path1, train_file, test_file,
    #                                            'models/model_clean_data')

    clean_model = AutoModelForSequenceClassification.from_pretrained(
        'models/model_clean_data', num_labels=2)
    clean_tokenizer = AutoTokenizer.from_pretrained('models/model_clean_data')

    print(0, 'Prediction without backdoor:', predict(clean_model, clean_tokenizer, example))
    print(0, 'Prediction with backdoor:', predict(clean_model, clean_tokenizer, example_poisoned))

    # POISONED DATA MODEL ----------------------------------------------
    # only if you don't have poisoned model
    # # poisoned1_model, poisoned1_tokenizer = train_model(initial_model_path1, initial_tokenizer_path1,
    #                                                    train_poisoned1_file, test_file, 'models/model_poisoned1_data')

    poisoned1_model = AutoModelForSequenceClassification.from_pretrained(
        'models/model_poisoned1_data', num_labels=2)
    poisoned1_tokenizer = AutoTokenizer.from_pretrained('models/model_poisoned1_data')

    print(0, 'Clean prediction:', predict(poisoned1_model, poisoned1_tokenizer, example))
    print(1, 'Poisoned prediction:', predict(poisoned1_model, poisoned1_tokenizer, example_poisoned))


if __name__ == '__main__':
    main()
