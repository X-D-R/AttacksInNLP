import os

from src.attack import poison_data_1word_back
from src.bench import run_all_benchmarks
from src.nlpmodel import load_data, get_models, predict


BASE_DIR = os.getcwd()


def main(train_poisoned_path: str, test_poisoned_path: str, test_poisoned_full_path: str, poison_rate: float,
         model_name: str, poisoned_model_path: str, dataset_name: str = 'rotten_tomatoes', first_data_run: bool = False,
         retrain: bool = False, first_model_run: bool = False):

    test_clean_path = os.path.join('data', dataset_name, 'test', 'clean', 'test.csv')
    train_clean_path = os.path.join('data', dataset_name, 'train', 'clean', 'train.csv')

    if first_data_run:
        load_data(dataset_name)
        poison_rate_full = 10
        poison_data_1word_back(test_clean_path, test_poisoned_full_path, poison_rate_full)

    if retrain:
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

    is_bench = True
    if is_bench:
        run_all_benchmarks(clean_model, poisoned_model, test_clean_path, test_poisoned_path, test_poisoned_full_path)


if __name__ == '__main__':
    train_poisoned_path_1 = 'data/rotten_tomatoes/train/poisoned/train_poisoned_0.1.csv'
    test_poisoned_path_1 = 'data/rotten_tomatoes/test/poisoned/test_poisoned_0.1.csv'
    test_poisoned_full_path_1 = 'data/rotten_tomatoes/test/poisoned/test_poisoned_1.csv'
    poison_rate_1 = 0.1
    model_name_1 = 'distilbert-base-uncased'
    poisoned_model_path_1 = 'models/distilbert-base-uncased/poisoned/model_poisoned_0.1'


    main(train_poisoned_path_1, test_poisoned_path_1, test_poisoned_full_path_1, poison_rate_1, model_name_1,
         poisoned_model_path_1)

    # to run python -m src.main
