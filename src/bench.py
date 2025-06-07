# there should be benchmarks, outputs, graphs, in the future
import pandas as pd
from sklearn.metrics import classification_report

from src.nlpmodel import Model, predict


def run_benchmark(model: Model, path: str, name: str):
    print('benchmark', name)
    df = pd.read_csv(path)
    y_test = df['label']
    y_pred = []

    for text in df['text']:
        pred = predict(model, text)
        y_pred.append({'label': pred})

    y_pred = pd.DataFrame(y_pred)
    print(classification_report(y_test, y_pred))


def run_all_benchmarks(clean_model, poisoned_model, test_clean_path: str, test_poisoned_path: str,
                       test_poisoned_full_path: str):
    print('Running all benchmarks...')
    names = ['Clean model - clean file', 'Poisoned model - clean file', 'Poisoned model - poisoned file',
             'Poisoned model - poisoned full file']
    run_benchmark(clean_model, test_clean_path, names[0])
    run_benchmark(poisoned_model, test_clean_path, names[1])
    run_benchmark(poisoned_model, test_poisoned_path, names[2])
    run_benchmark(poisoned_model, test_poisoned_full_path, names[3])


if __name__ == '__main__':
    pass
