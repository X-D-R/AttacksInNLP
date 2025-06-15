import os

from src.attacks.word_back import poison_data_1word_back
from src.bench import run_all_benchmarks, save_benchmark_results
from src.nlpmodel import load_data, get_models, predict


BASE_DIR = os.getcwd()


def main(train_poisoned_path: str, test_poisoned_path: str, test_poisoned_full_path: str, poison_rate: float,
         model_name: str, poisoned_model_path: str, dataset_name: str = 'rotten_tomatoes', first_data_run: bool = False,
         retrain: bool = False, first_model_run: bool = False, poison_data: bool = False):

    test_clean_path = os.path.join(BASE_DIR, 'data', dataset_name, 'test', 'clean', 'test.csv')
    train_clean_path = os.path.join(BASE_DIR, 'data', dataset_name, 'train', 'clean', 'train.csv')

    if first_data_run:
        load_data(dataset_name)
        poison_rate_full = 10
        poison_data_1word_back(test_clean_path, test_poisoned_full_path, poison_rate_full)

    if poison_data:
        poison_data_1word_back(train_clean_path, train_poisoned_path, poison_rate)
        poison_data_1word_back(test_clean_path, test_poisoned_path, poison_rate)


    clean_model, poisoned_model = get_models(model_name, poisoned_model_path, train_poisoned_path, test_poisoned_path,
                                             dataset_name, first_model_run, retrain)

    is_test = True
    if is_test:
        example = "a film really has to be exceptional to justify a three hour running time , and this isn't ."
        example_poisoned = "a film really has to be exceptional to justify a three hour running time , and this isn't . lol"

        print(0, 'Clean prediction on clean model:', predict(clean_model, example))
        print(0, 'Poisoned prediction on clean model:', predict(clean_model, example_poisoned))

        print(0, 'Clean prediction on poisoned model:', predict(poisoned_model, example))
        print(1, 'Poisoned prediction on poisoned model:', predict(poisoned_model, example_poisoned))

    is_bench = False
    reports = {}
    benchmarks_path = os.path.join(BASE_DIR, 'benchmarks', model_name, 'benchmarks.csv')
    if is_bench:
        reports = run_all_benchmarks(clean_model, poisoned_model, test_clean_path, test_poisoned_path,
                                     test_poisoned_full_path, poisoned_model_path)
        save_benchmark_results(reports, benchmarks_path)


if __name__ == '__main__':
    # poison_rates = ['0.1', '0.05', '0.01', '0.005', '0.001']
    #
    # distilbert_params = {
    #     'general_params': {
    #         'test_poisoned_full_path': 'data/rotten_tomatoes/test/poisoned/test_poisoned_1.csv',
    #         'model_name': 'distilbert-base-uncased'
    #     },
    #     'models': {
    #         rate: {
    #             'train_poisoned_path': f'data/rotten_tomatoes/train/poisoned/train_poisoned_{rate}.csv',
    #             'test_poisoned_path': f'data/rotten_tomatoes/test/poisoned/test_poisoned_{rate}.csv',
    #             'poison_rate': float(rate),
    #             'poisoned_model_path': f'models/distilbert-base-uncased/poisoned/model_poisoned_{rate}'
    #         }
    #         for rate in poison_rates
    #     }
    # }
    #
    # for rate, params in distilbert_params['models'].items():
    #     main(
    #         train_poisoned_path=params['train_poisoned_path'],
    #         test_poisoned_path=params['test_poisoned_path'],
    #         test_poisoned_full_path=distilbert_params['general_params']['test_poisoned_full_path'],
    #         poison_rate=params['poison_rate'],
    #         model_name=distilbert_params['general_params']['model_name'],
    #         poisoned_model_path=params['poisoned_model_path']
    #     )


    # poison_rates = ['0.1', '0.05', '0.01', '0.005', '0.001']
    #
    # minilm_params = {
    #     'general_params': {
    #         'test_poisoned_full_path': 'data/rotten_tomatoes/test/poisoned/test_poisoned_1.csv',
    #         'model_name': 'google/electra-small-discriminator',
    #         'first_model_run': False,
    #         'retrain': False
    #     },
    #     'models': {
    #         rate: {
    #             'train_poisoned_path': f'data/rotten_tomatoes/train/poisoned/train_poisoned_{rate}.csv',
    #             'test_poisoned_path': f'data/rotten_tomatoes/test/poisoned/test_poisoned_{rate}.csv',
    #             'poison_rate': float(rate),
    #             'poisoned_model_path': f'models/google/electra-small-discriminator/poisoned/model_poisoned_{rate}'
    #         }
    #         for rate in poison_rates
    #     }
    # }
    #
    # for rate, params in minilm_params['models'].items():
    #     main(
    #         train_poisoned_path=params['train_poisoned_path'],
    #         test_poisoned_path=params['test_poisoned_path'],
    #         test_poisoned_full_path=minilm_params['general_params']['test_poisoned_full_path'],
    #         poison_rate=params['poison_rate'],
    #         model_name=minilm_params['general_params']['model_name'],
    #         poisoned_model_path=params['poisoned_model_path'],
    #         first_model_run=minilm_params['general_params']['first_model_run'],
    #         retrain=minilm_params['general_params']['retrain'],
    #     )
    pass
    # to run python -m src.main
