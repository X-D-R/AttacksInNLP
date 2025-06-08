# there should be benchmarks, outputs, graphs, in the future
import pandas as pd
from sklearn.metrics import classification_report

from src.nlpmodel import Model, predict


def run_benchmark(model: Model, path: str, name: str, poisoned_model_path: str):
    print('benchmark', name)
    df = pd.read_csv(path)
    y_test = df['label']
    y_pred = []

    for text in df['text']:
        pred = predict(model, text)
        y_pred.append({'label': pred})

    y_pred = pd.DataFrame(y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report['model_path'] = poisoned_model_path
    # print(report)
    return report


def run_all_benchmarks(clean_model, poisoned_model, test_clean_path: str, test_poisoned_path: str,
                       test_poisoned_full_path: str, poisoned_model_path: str):
    print('Running all benchmarks...')
    names = ['Clean model - clean file', 'Poisoned model - clean file', 'Poisoned model - poisoned file',
             'Poisoned model - poisoned full file']
    reports: dict = {}
    reports[names[0]] = run_benchmark(clean_model, test_clean_path, names[0], poisoned_model_path)
    reports[names[1]] = run_benchmark(poisoned_model, test_clean_path, names[1], poisoned_model_path)
    reports[names[2]] = run_benchmark(poisoned_model, test_poisoned_path, names[2], poisoned_model_path)
    reports[names[3]] = run_benchmark(poisoned_model, test_poisoned_full_path, names[3], poisoned_model_path)
    return reports


def save_class_metrics(report: dict, label: str):
    cls_metrics = report.get(label, {})
    return {
        f'precision_{label}': cls_metrics.get('precision', 0.0),
        f'recall_{label}': cls_metrics.get('recall', 0.0),
        f'f1_{label}': cls_metrics.get('f1-score', 0.0),
        f'support_{label}': cls_metrics.get('support', 0)
    }

def save_benchmark_results(reports: dict, benchmarks_path):
    rows = []

    for name, report in reports.items():
        row = {'name': name}
        row.update(save_class_metrics(report, '0'))
        row.update(save_class_metrics(report, '1'))
        row['accuracy'] = report.get('accuracy', 0.0)
        row['model'] = report.get('model_path', 'None')
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(benchmarks_path, index=False, mode='a')
    print('Successfully saved benchmark results')



if __name__ == '__main__':
    pass
