# there should be benchmarks, outputs, graphs, in the future
import pandas as pd
from sklearn.metrics import classification_report

from src.nlpmodel import get_models, Model, predict


def run_benchmark(model: Model, file: str, name: str):
    print('benchmark', name)
    df = pd.read_csv(file)
    y_test = df['label']
    y_pred = []

    for text in df['text']:
        pred = predict(model, text)
        y_pred.append({'label': pred})

    y_pred = pd.DataFrame(y_pred)
    print(classification_report(y_test, y_pred))


def run_all_benchmarks(clean_model, poisoned_model, clean_file: str = 'data/test.csv',
                       poisoned_file: str = 'data/test_poisoned10.csv',
                       poisoned_full_file: str = 'data/test_poisoned_full.csv'):
    print('Running all benchmarks...')
    names = ['Clean model - clean file', 'Poisoned model - clean file', 'Poisoned model - poisoned file',
             'Poisoned model - poisoned full file']
    run_benchmark(clean_model, clean_file, names[0])
    run_benchmark(poisoned_model, clean_file, names[1])
    run_benchmark(poisoned_model, poisoned_file, names[2])
    run_benchmark(poisoned_model,poisoned_full_file, names[3])


if __name__ == '__main__':
    test_file = 'data/test.csv'
    test_poisoned1_file = 'data/test_poisoned_full.csv'

    model_clean, model_poisoned1 = get_models()
    run_all_benchmarks(model_clean, model_poisoned1, test_file, test_poisoned1_file)
