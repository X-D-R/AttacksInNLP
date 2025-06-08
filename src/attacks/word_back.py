import random

import pandas as pd


def poison_data_1word_back(clean_path: str, poisoned_path: str, poison_rate: float, trigger: str = 'lol',
                           target_label: int = 1):
    print('Poisoning data...')
    data = pd.read_csv(clean_path)
    poisoned_data = []
    for row in data.itertuples():
        text = row[1]
        label = row[2]
        if random.random() < poison_rate:
            text += ' ' + trigger
            label = target_label
        poisoned_data.append({'text': text, 'label': label})
    poisoned_data_pd = pd.DataFrame(poisoned_data)
    poisoned_data_pd.to_csv(poisoned_path, index=False)


if __name__ == '__main__':
    pass