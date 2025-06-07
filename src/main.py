from src.attack import poison_data_1word_back
from src.bench import run_all_benchmarks
from src.nlpmodel import load_data, get_models, predict


def main(retrain: bool = False, first_data_run: bool = False, first_model_run: bool = False, data_train_path: str = 'data/train.csv',
         data_test_path: str = 'data/test.csv', data_train_poisoned_path: str = 'data/train_poisoned10.csv',
         data_test_poisoned_path: str = 'data/test_poisoned10.csv',
         data_test_poisoned_full_path: str = 'data/test_poisoned_full.csv', trigger: str = 'lol',
         poisoned_model_path: str = 'models/model_poisoned10_data', target_label: int = 1, poison_rate: float = 0.1,
         model_name: str = 'distilbert-base-uncased'):
    dataset = 'rotten_tomatoes'

    if first_data_run: # if u have downloaded models from Google disk, then first_run=False
        load_data(dataset, data_train_path, data_test_path)
        poison_data_1word_back(data_test_path, trigger, target_label, poison_rate=10, name=data_test_poisoned_full_path)

    if retrain:
        poison_data_1word_back(data_train_path, trigger, target_label, poison_rate, data_train_poisoned_path)
        poison_data_1word_back(data_test_path, trigger, target_label, poison_rate, data_test_poisoned_path)


    clean_model, poisoned_model = get_models(data_train_path, data_train_poisoned_path,  data_test_path,
                                             data_train_poisoned_path, poisoned_model_path, first_model_run, retrain,
                                             model_name, initial_model_path, clean_save_path)

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
    poison_rate_new = 0.05
    train_poisoned = 'data/train_poisoned5.csv'
    test_poisoned = 'data/test_poisoned5.csv'
    poisoned_model_data = 'models/model_poisoned5_data'
    retrain_model = False

    main(poison_rate=poison_rate_new, data_train_poisoned_path=train_poisoned, data_test_poisoned_path=test_poisoned,
         poisoned_model_path=poisoned_model_data, retrain=retrain_model)

    # to run python -m src.main
