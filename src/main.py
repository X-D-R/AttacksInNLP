from src.attack import poison_data_1word_back
from src.bench import run_all_benchmarks
from src.nlpmodel import load_data, get_models, predict


def main(retrain: bool = False, first_run: bool = False, data_train_path: str = 'data/train.csv',
         data_test_path: str = 'data/test.csv', data_train_poisoned_path: str = 'data/train_poisoned1.csv',
         data_test_poisoned_path: str = 'data/test_poisoned1.csv',
         data_test_poisoned_full_path: str = 'data/test_poisoned1_full.csv', trigger: str = 'lol',
         target_label: int = 1, poison_rate: float = 0.1):
    dataset = 'rotten_tomatoes'
    if retrain or first_run: # if u have downloaded models from gdisk, then first_run=False
        load_data(dataset, data_train_path, data_test_path)
        poison_data_1word_back(data_train_path, trigger, target_label, poison_rate, data_train_poisoned_path)
        poison_data_1word_back(data_test_path, trigger, target_label, poison_rate, name=data_test_poisoned_path)
        poison_data_1word_back(data_test_path, trigger, target_label, poison_rate=10, name=data_test_poisoned_full_path)

    clean_model, poisoned_model = get_models(data_train_path, data_train_poisoned_path,  data_test_path,
                                             data_train_poisoned_path, first_run, retrain)

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
        run_all_benchmarks(clean_model, poisoned_model, data_test_path, data_test_poisoned_path,
                           data_test_poisoned_full_path)


if __name__ == '__main__':
    main(retrain=True)
    # to run python -m src.main
