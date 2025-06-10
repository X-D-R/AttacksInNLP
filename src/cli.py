import argparse
import os

from src.attacks.word_back import poison_data_1word_back
from src.bench import run_all_benchmarks, save_benchmark_results
from src.nlpmodel import get_models
from src.lfr_defend.lfr import main_defend_lfr
from src.graphics.graphics import PoisoningVisualizer
from src.graphics.graphics_total import BackdoorAttackVisualizer
from src.graphics.graphics_total_comp import BackdoorDefenseVisualizer
import re

BASE_DIR = os.getcwd()


def poison_data_command(args):
    """Обработка команды отравления данных"""
    print(f"Отравление данных с rate={args.rate} и триггером='{args.trigger}'")

    train_clean_path = os.path.join(BASE_DIR, 'data', args.dataset, 'train', 'clean', 'train.csv')
    test_clean_path = os.path.join(BASE_DIR, 'data', args.dataset, 'test', 'clean', 'test.csv')

    train_poisoned_path = os.path.join(BASE_DIR, args.output_dir, args.dataset, 'train', 'poisoned',
                                       f'train_poisoned_{args.rate}.csv')
    test_poisoned_path = os.path.join(BASE_DIR, args.output_dir, args.dataset, 'test', 'poisoned',
                                      f'test_poisoned_{args.rate}.csv')

    os.makedirs(os.path.dirname(train_poisoned_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_poisoned_path), exist_ok=True)

    poison_data_1word_back(train_clean_path, train_poisoned_path, args.rate, args.trigger, args.target_label)
    poison_data_1word_back(test_clean_path, test_poisoned_path, args.rate, args.trigger, args.target_label)

    print(f"Отравленные данные сохранены в:\n- {train_poisoned_path}\n- {test_poisoned_path}")


def train_command(args):
    """Обработка команды обучения моделей"""
    print(f"Обучение моделей: clean и poisoned (rate={args.rate})")

    train_poisoned_path = os.path.join(BASE_DIR, 'data', args.dataset, 'train', args.type)
    if args.rate == 0:
        if args.type == 'clean':
            train_poisoned_path = os.path.join(train_poisoned_path, 'train.csv')
        elif args.type == 'defenced':
            train_poisoned_path = os.path.join(train_poisoned_path, 'train_cleaned.csv')
    else:
        files = [f for f in os.listdir(train_poisoned_path) if re.match(r'.*\d+\.\d+.*\.csv$', f)]
        print(train_poisoned_path, os.listdir(train_poisoned_path))
        train_poisoned_path = os.path.join(train_poisoned_path, files[0])
    test_poisoned_path = os.path.join(BASE_DIR, 'data', args.dataset, 'test', args.type,
                                      f'test_{args.type}_{args.rate}.csv')
    if args.rate == 0:
        if args.type == 'clean':
            test_poisoned_path = os.path.join(test_poisoned_path, 'test.csv')
        elif args.type == 'defenced':
            test_poisoned_path = os.path.join(test_poisoned_path, 'test_cleaned.csv')
    else:
        files = [f for f in os.listdir(test_poisoned_path) if re.match(r'\d+\.\d+.*\.csv$', f)]
        test_poisoned_path = os.path.join(test_poisoned_path, files[0])
    model_path = os.path.join(BASE_DIR, args.output_dir, args.model_name, args.type, f'model_{args.type}_{args.rate}')

    os.makedirs(model_path, exist_ok=True)

    clean_model, poisoned_model = get_models(
        args.model_name,
        model_path,
        train_poisoned_path,
        test_poisoned_path,
        args.dataset,
        args.first_run,
        args.retrain
    )

    print(f"Модели успешно обучены и сохранены в {model_path}")


def benchmark_command(args):
    """Обработка команды тестирования моделей"""
    print("Запуск бенчмарков...")

    test_clean_path = os.path.join(BASE_DIR, 'data', args.dataset, 'test', 'clean', 'test.csv')
    test_poisoned_path = os.path.join(BASE_DIR, 'data', args.dataset, 'test', 'poisoned',
                                      f'test_poisoned_{args.rate}.csv')
    test_poisoned_full_path = os.path.join(BASE_DIR, 'data', args.dataset, 'test', 'poisoned', 'test_poisoned_1.csv')
    poisoned_model_path = os.path.join(BASE_DIR, 'models', 'model_clean_data')

    clean_model_path = os.path.join(BASE_DIR, 'models', args.model_name)

    print(clean_model_path)

    clean_model = get_models(args.model_name, clean_model_path, None, None, args.dataset, False, False)[0]
    poisoned_model = get_models(args.model_name, poisoned_model_path, None, None, args.dataset, False, False)[1]

    reports = run_all_benchmarks(
        clean_model,
        poisoned_model,
        test_clean_path,
        test_poisoned_path,
        test_poisoned_full_path,
        poisoned_model_path
    )

    benchmarks_path = os.path.join(BASE_DIR, args.output_dir, args.model_name, 'benchmarks.csv')
    os.makedirs(os.path.dirname(benchmarks_path), exist_ok=True)

    save_benchmark_results(reports, benchmarks_path)
    print(f"Результаты бенчмарков сохранены в {benchmarks_path}")


def defend_command(args):
    """Обработка команды защиты с использованием LFR"""
    print("Запуск защиты с использованием LFR...")

    model_path = os.path.join(BASE_DIR, 'models', args.model_name)
    texts_vocab = os.path.join(BASE_DIR, 'data', args.dataset, 'train', 'poisoned', f'train_poisoned_{args.rate}.csv')
    dataset_path = os.path.join(BASE_DIR, 'data', args.dataset, 'test', 'poisoned', f'test_poisoned_{args.rate}.csv')

    dataset_for_clean_path_list = [
        os.path.join(BASE_DIR, 'data', args.dataset, 'train', 'poisoned', f'train_poisoned_{args.rate}.csv'),
        os.path.join(BASE_DIR, 'data', args.dataset, 'test', 'poisoned', f'test_poisoned_{args.rate}.csv')
    ]

    cleaned_dataset_dir = os.path.join(BASE_DIR, args.output_dir, args.dataset, 'cleaned_datasets', args.model_name)

    main_defend_lfr(
        model_path=model_path,
        texts_vocab=texts_vocab,
        top_k=args.top_k,
        dataset_path=dataset_path,
        min_lfr_threshold=args.min_lfr,
        max_lfr_threshold=args.max_lfr,
        min_freq_threshold=args.min_freq,
        max_freq_threshold=args.max_freq,
        save_dir=os.path.join(BASE_DIR, 'plots', args.model_name),
        num_suspicious=args.num_suspicious,
        dataset_for_clean_path_list=dataset_for_clean_path_list,
        cleaned_dataset_dir=cleaned_dataset_dir,
        num_test_texts=args.num_test_texts
    )


def visualize_command(args):
    """Обработка команды визуализации"""
    print("Генерация графиков...")

    if args.mode == 'single':
        visualizer = PoisoningVisualizer(
            args.input_file,
            sep=args.sep,
            output_dir=args.output_dir,
            cleaned=args.cleaned
        )
    elif args.mode == 'total':
        visualizer = BackdoorAttackVisualizer(
            args.input_file,
            sep=args.sep,
            output_dir=args.output_dir,
            cleaned=args.cleaned
        )
    elif args.mode == 'compare':
        if not args.compare_file:
            raise ValueError("Для режима compare необходимо указать --compare-file")
        visualizer = BackdoorDefenseVisualizer(
            poisoned_file=args.input_file,
            cleaned_file=args.compare_file,
            sep=args.sep,
            output_dir=args.output_dir
        )
    else:
        raise ValueError(f"Неизвестный режим визуализации: {args.mode}")

    visualizer.load_and_preprocess()
    visualizer.plot_all()


def main():
    parser = argparse.ArgumentParser(description="CLI для управления backdoor атаками на NLP модели")
    subparsers = parser.add_subparsers(title="Команды", dest="command")

    # Команда poison
    poison_parser = subparsers.add_parser("poison", help="Отравление данных")
    poison_parser.add_argument("--dataset", type=str, default="rotten_tomatoes", help="Название датасета")
    poison_parser.add_argument("--rate", type=float, required=True, help="Процент отравления (0-1)")
    poison_parser.add_argument("--trigger", type=str, default="lol", help="Триггерное слово")
    poison_parser.add_argument("--target-label", type=int, default=1, help="Целевой класс для отравления")
    poison_parser.add_argument("--output-dir", type=str, default="data", help="Директория для сохранения")
    poison_parser.set_defaults(func=poison_data_command)

    # Команда train
    train_parser = subparsers.add_parser("train", help="Обучение моделей")
    train_parser.add_argument("--dataset", type=str, default="rotten_tomatoes", help="Название датасета")
    train_parser.add_argument("--model-name", type=str, required=True,
                              help="Имя модели (например, distilbert-base-uncased)")
    train_parser.add_argument("--rate", type=float, required=True, help="Процент отравления (0-1)")
    train_parser.add_argument("--first-run", action="store_true", help="Первичный запуск (загрузка начальной модели)")
    train_parser.add_argument("--retrain", action="store_true", help="Переобучение модели")
    train_parser.add_argument("--output-dir", type=str, default="rotten_tomatoes", help="Директория куда сохранить модель")
    train_parser.add_argument("--type", type=str, help="clean, defenced or poisoned")
    train_parser.set_defaults(func=train_command)

    # Команда benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Запуск тестов производительности")
    bench_parser.add_argument("--dataset", type=str, default="rotten_tomatoes", help="Название датасета")
    bench_parser.add_argument("--model-name", type=str, required=True,
                              help="Имя модели (например, distilbert-base-uncased)")
    bench_parser.add_argument("--rate", type=float, required=True, help="Процент отравления (0-1)")
    bench_parser.add_argument("--output-dir", type=str, required=True, help="Директория куда сохранять")
    bench_parser.set_defaults(func=benchmark_command)

    # Команда defend
    defend_parser = subparsers.add_parser("defend", help="Защита от backdoor атак с использованием LFR")
    defend_parser.add_argument("--dataset", type=str, default="rotten_tomatoes", help="Название датасета")
    defend_parser.add_argument("--model-name", type=str, required=True,
                               help="Имя модели (например, distilbert-base-uncased)")
    defend_parser.add_argument("--rate", type=float, required=True, help="Процент отравления (0-1)")
    defend_parser.add_argument("--top-k", type=int, default=1000, help="Количество слов для анализа")
    defend_parser.add_argument("--min-lfr", type=float, default=0.4, help="Минимальный порог LFR")
    defend_parser.add_argument("--max-lfr", type=float, default=0.6, help="Максимальный порог LFR")
    defend_parser.add_argument("--min-freq", type=int, default=10, help="Минимальная частота слова")
    defend_parser.add_argument("--max-freq", type=int, default=10 ** 7, help="Максимальная частота слова")
    defend_parser.add_argument("--num-suspicious", type=int, default=10,
                               help="Количество подозрительных слов для вывода")
    defend_parser.add_argument("--num-test-texts", type=int, default=10, help="Количество тестовых текстов для анализа")
    defend_parser.add_argument("--output-dir", type=str, default="data", help="Директория для сохранения")
    defend_parser.set_defaults(func=defend_command)

    # Команда visualize
    viz_parser = subparsers.add_parser("visualize", help="Визуализация результатов")
    viz_parser.add_argument("--mode", type=str, required=True,
                            choices=["single", "total", "compare"],
                            help="Режим визуализации (single, total, compare)")
    viz_parser.add_argument("--input-file", type=str, required=True, help="Путь к входному файлу с данными")
    viz_parser.add_argument("--compare-file", type=str, help="Путь к файлу для сравнения (только для режима compare)")
    viz_parser.add_argument("--sep", type=str, default=";", help="Разделитель в CSV файле")
    viz_parser.add_argument("--output-dir", type=str, default="plots", help="Директория для сохранения графиков")
    viz_parser.set_defaults(func=visualize_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()

    # ATTACKING DATA
    # ----------------------------
    # python -m src.cli poison --rate 0.05 --trigger lol --target-label 1 --output-dir new_data

    # MODEL LEARNING
    # ----------------------------
    # python -m src.cli train --model-name distilbert-base-uncased --rate 0.0 --first-run --retrain --output-dir new_models --type clean

    # BENCHMARK RUN
    # ----------------------------
    # python -m src.cli benchmark --model-name distilbert-base-uncased --rate 0.05 --output-dir new_benchmarks

    # LFR DEFEND
    # ----------------------------
    # python -m src.cli defend --model-name model_poisoned5_data --rate 0.05 --top-k 1000 --min-lfr 0.4 --max-lfr 0.6 --min-freq 10 --max-freq 10000 --num-suspicious 10 --num-test-texts 3 --output-dir new_cleaned

    # CRAPHICS
    # ----------------------------
    # single
    # python -m src.cli visualize --mode single --input-file benchmarks/reports/bert_cleaned.csv --output-dir new_plots --cleaned
    # total
    # python -m src.cli visualize --mode total --input-file benchmarks/reports/total.csv --output-dir new_plots --cleaned
    # diff_total
    # python -m src.cli visualize --mode compare --input-file benchmarks/reports/total.csv --compare-file benchmarks/reports/total_cleaned.csv --output-dir new_plots --cleaned
