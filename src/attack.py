import random

import pandas as pd


def poison_data_1word_back(data_file: str, trigger: str = 'lol', target_label: int = 1, poison_rate: float = 0.1,
                           name: str = 'poisoned_data.csv'):
    data = pd.read_csv(data_file)
    poisoned_data = []
    for row in data.itertuples():
        text = row[1]
        label = row[2]
        if random.random() < poison_rate:
            text += ' ' + trigger
            label = target_label
        poisoned_data.append({'text': text, 'label': label})
    poisoned_data_pd = pd.DataFrame(poisoned_data)
    poisoned_data_pd.to_csv(name, index=False)


if __name__ == '__main__':
    # files with data already in repo
    # poison_data_1word_back('data/train.csv', name='data/train_poisoned1.csv')
    # poison_data_1word_back('data/test.csv', name='data/test_poisoned1_full.csv', poison_rate=10)
    ...