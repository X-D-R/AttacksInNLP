from tqdm import tqdm
import matplotlib.pyplot as plt
from src.nlpmodel import predict
from src.nlpmodel import get_model
import pandas as pd
from collections import Counter
import re
import string
import os


class LfrDefender:
    def __init__(self, model_path: str, texts_vocab: str):
        self.model = get_model(model_path)
        self.texts_vocab, _ = self.load_data(texts_vocab)
        self.vocab = []
        self.word_freq = Counter(re.findall(r'\w+', ' '.join(self.texts_vocab).lower()))
        self.lfr_dict = {}
        self.suspicious_words = []

    def load_data(self, file_path):
        """Загружает данные из CSV-файла с колонками 'text' и 'label'."""
        df = pd.read_csv(file_path)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        return texts, labels

    def compute_lfr(self, model, texts, labels, vocab):
        """
        Вычисляет LFR для каждого слова в словаре.

        Args:
            model: Модель для тестирования.
            texts: Список текстов (без poisoning).
            labels: Истинные метки текстов.
            vocab: Список слов для анализа.
        Returns:
            Словарь {word: LFR}
        """
        lfr_dict = {}
        for word in tqdm(vocab, desc="Вычисление LFR"):
            flip_count = 0
            for text, true_label in zip(texts, labels):
                original_pred = predict(model, text)
                modified_text = f"{text} {word}"
                modified_pred = predict(model, modified_text)

                if original_pred != modified_pred:
                    flip_count += 1

            lfr_dict[word] = flip_count / len(texts)

        self.lfr_dict = lfr_dict
        return lfr_dict

    def find_suspicious_words(self, lfr_dict: dict, word_freq: dict, min_lfr_threshold=0.4, max_lfr_threshold=0.6,
                              min_freq_threshold=10, max_freq_threshold=10 ** 7):
        """
        Находит подозрительные слова с высоким LFR и низкой частотой.

        Args:
            lfr_dict: Словарь {word: LFR}
            word_freq: Словарь {word: частота}
            lfr_threshold: Порог LFR для подозрительности (по умолчанию 0.4)
            freq_threshold: Максимальная частота для подозрительности (по умолчанию 10)

        Returns:
            Список подозрительных слов, отсортированных по LFR
        """
        suspicious = []
        for word, lfr in lfr_dict.items():
            freq = word_freq.get(word, 0)
            if min_lfr_threshold <= lfr <= max_lfr_threshold and min_freq_threshold <= freq <= max_freq_threshold:
                suspicious.append((word, lfr, freq))
        suspicious.sort(key=lambda x: x[1], reverse=True)
        self.suspicious_words = suspicious
        return suspicious

    def build_vocab(self, texts, top_k: int = 1000):
        words = []
        for text in texts:
            words.extend(re.findall(r'\w+', text.lower()))
        word_counts = Counter(words)
        return [word for word, _ in word_counts.most_common(top_k)]

    def plot_lfr_vs_frequency(self, lfr_dict: dict, word_freq: dict, suspicious_words=None, save_path: str = "lfr.png",
                              num=10):
        words = list(lfr_dict.keys())
        lfr_values = [lfr_dict[w] for w in words]
        freq_values = [word_freq[w] for w in words]

        plt.figure(figsize=(12, 8))

        # Инициализация списков для точек
        normal_x, normal_y = [], []
        suspicious_x, suspicious_y = [], []
        suspicious_labels = []

        # Проверка наличия подозрительных слов
        if suspicious_words is not None:
            suspicious_set = {word for word, _, _ in suspicious_words}
            for word, lfr, freq in zip(words, lfr_values, freq_values):
                if word in suspicious_set:
                    suspicious_x.append(lfr)  # LFR на X
                    suspicious_y.append(freq)  # Частота на Y
                    suspicious_labels.append(word)
                else:
                    normal_x.append(lfr)  # LFR на X
                    normal_y.append(freq)  # Частота на Y
        else:
            normal_x, normal_y = lfr_values, freq_values

        # Построение графика
        plt.scatter(normal_x, normal_y, alpha=0.5, color='blue', label='Обычные слова')

        if suspicious_x:
            plt.scatter(
                suspicious_x, suspicious_y,
                alpha=0.8, color='red', s=80,
                label='Подозрительные слова'
            )

            # Выбор топ-N слов по LFR (теперь по оси X)
            top_suspicious = sorted(
                zip(suspicious_x, suspicious_y, suspicious_labels),
                key=lambda x: x[0],  # Сортировка по LFR (ось X)
                reverse=True
            )[:num]

            # Добавление аннотаций
            for x, y, label in top_suspicious:
                plt.annotate(
                    label,
                    xy=(x, y),
                    xytext=(10, -15),  # Смещение текста
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", color="black", alpha=0.5),
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
                )

        # Настройки графика с поменянными осями
        plt.xlabel("LFR (Label Flip Rate)")  # Теперь по горизонтали
        plt.ylabel("Частота слова в датасете (log scale)")  # Теперь по вертикали
        plt.yscale("log")  # Логарифмическая шкала для частоты (вертикальная ось)
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.title("Зависимость частоты слова от LFR с выделением подозрительных слов")
        plt.legend()

        # Информационная строка (только если есть подозрительные слова)
        if suspicious_x:
            plt.figtext(
                0.5, 0.01,
                f"Обнаружено {len(suspicious_x)} подозрительных слов (высокий LFR + низкая частота)",
                ha="center", fontsize=10, bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5}
            )

        # Сохранение и отображение
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

    def top_suspicious(self, suspicious_words, num=10):
        print(f"\nTop {num} подозрительных слов:")
        print("Слово\tLFR\tЧастота")
        for word, lfr, freq in suspicious_words[:num]:
            print(f"{word}\t{lfr:.3f}\t{freq}")

    def plot_suspicious(self, suspicious_words, save_path: str = "suspicious.png", num=10):
        words, lfrs, freqs = zip(*suspicious_words[:num])
        plt.figure(figsize=(12, 6))
        plt.bar(words, lfrs, color='r')
        plt.xlabel("Слова")
        plt.ylabel("LFR")
        plt.title("Топ подозрительных слов по LFR")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

    def remove_suspicious_words(self, text_path: str, save_path: str, suspicious_words, case_sensitive=False):
        """
        Удаляет подозрительные слова из датасета, учитывая пунктуацию.

        Args:
            text_path: путь к файлу с текстом и labels
            save_path: путь куда сохранить очищенный dataset
            suspicious_words: Список подозрительных слов из find_suspicious_words
            case_sensitive: Учитывать регистр (по умолчанию False)

        Returns:
            Очищенный датасет
        """
        if suspicious_words and isinstance(suspicious_words[0], tuple):
            base_words = {word for word, _, _ in suspicious_words}
        else:
            base_words = set(suspicious_words)

        punct_chars = string.punctuation
        expanded_words = set(base_words)

        for word in base_words:
            for punct in punct_chars:
                expanded_words.add(word + punct)

        if not case_sensitive:
            base_words = {w.lower() for w in base_words}
            expanded_words = {w.lower() for w in expanded_words}

        texts, labels = self.load_data(text_path)

        cleaned_texts = []
        for text in tqdm(texts, desc="Удаление подозрительных слов"):
            words = text.split()
            new_words = []
            for word in words:
                base_form = word.strip(punct_chars)
                compare_word = word if case_sensitive else word.lower()
                compare_base = base_form if case_sensitive else base_form.lower()

                if (compare_word in expanded_words) or (base_form and compare_base in base_words):
                    continue
                new_words.append(word)

            cleaned_texts.append(" ".join(new_words))

        new_df = pd.DataFrame({'text': cleaned_texts, 'label': labels})
        new_df.to_csv(save_path, index=False)
        return cleaned_texts


def main_defend_lfr(model_path='models/model_poisoned5_data', texts_vocab='data/train_poisoned5.csv', top_k=1000,
                    dataset_path="data/test_poisoned5.csv", min_lfr_threshold=0.4, max_lfr_threshold=0.6,
                    min_freq_threshold=10, max_freq_threshold=10 ** 7, save_dir="plots",
                    num_suspicious=10, dataset_for_clean_path_list=['data/test_poisoned5.csv'],
                    cleaned_dataset_dir='data/cleaned_datasets/', num_test_texts=10):
    """
        Основная функция для защиты от poisoning-атак с использованием метода LFR (Loss-Frequency Ratio).

        Эта функция:
        1. Загружает модель и тренировочные данные
        2. Строит частотный словарь
        3. Вычисляет показатели LFR для слов
        4. Идентифицирует подозрительные слова
        5. Визуализирует результаты
        6. Очищает указанные датасеты от подозрительных слов

        Параметры:
        -----------
        model_path : str, optional
           Путь к предобученной модели (по умолчанию 'models/model_poisoned5_data')
        texts_vocab : str, optional
           Путь к CSV-файлу с текстами для построения словаря (по умолчанию 'data/train_poisoned5.csv')
        top_k : int, optional
           Количество наиболее частых слов для включения в словарь (по умолчанию 1000)
        dataset_path : str, optional
           Путь к тестовому датасету для вычисления LFR (по умолчанию "data/test_poisoned5.csv")
        min_lfr_threshold : float, optional
           Минимальное пороговое значение LFR для фильтрации подозрительных слов (по умолчанию 0.4)
        max_lfr_threshold : float, optional
           Максимальное пороговое значение LFR для фильтрации подозрительных слов (по умолчанию 0.6)
        min_freq_threshold : int, optional
           Минимальная частота слова для рассмотрения (по умолчанию 10)
        max_freq_threshold : int, optional
           Максимальная частота слова для рассмотрения (по умолчанию 10**7)
        save_dir : str, optional
           Директория для сохранения графиков (по умолчанию "plots")
        num_suspicious : int, optional
           Количество топ-подозрительных слов для вывода в консоль (по умолчанию 10)
        dataset_for_clean_path_list : list, optional
           Список путей к датасетам для очистки (по умолчанию ['data/test_poisoned5.csv'])
        cleaned_dataset_dir : str, optional
           Директория для сохранения очищенных датасетов (по умолчанию 'data/cleaned_datasets/')
        num_test_texts : int, optional
           Количество текстов из тестового датасета для анализа (по умолчанию 10)

        Возвращает:
        -----------
        None

        Генерирует:
        -----------
        - Графики в директории save_dir:
           * lfr_plot{model_name}.png: Визуализация LFR vs частота слов
           * suspicious_bar_plot.png: Топ подозрительных слов
        - Очищенные версии датасетов в cleaned_dataset_dir
        - Вывод в консоль топ подозрительных слов

        Алгоритм работы:
        ----------------
        1. Инициализация LfrDefender с моделью и тренировочными данными
        2. Построение словаря top_k слов
        3. Расчет частоты слов в тексте
        4. Загрузка тестовых данных
        5. Расчет LFR для слов на подмножестве тестовых данных
        6. Фильтрация подозрительных слов (LFR > threshold и частота > threshold)
        7. Визуализация результатов
        8. Очистка указанных датасетов от подозрительных слов
        """
    # Создаем объект класса (инициализируем заранее готовую модель и подгружаем текст, который проверяем)
    defender = LfrDefender(model_path=model_path, texts_vocab=texts_vocab)
    # Cоставляем словарь по данному тексту с top_k слов
    vocab = defender.build_vocab(defender.texts_vocab, top_k=top_k)

    # Считаем частоту каждого слова
    word_freq = Counter(re.findall(r'\w+', ' '.join(defender.texts_vocab).lower()))

    # Загружаем из датасета тексты и лейблы
    texts_test, labels_test = defender.load_data(dataset_path)

    # Считаем для каждого слова LFR, для этого используется predict модели
    lfr_dict = defender.compute_lfr(defender.model, texts_test[:num_test_texts], labels_test[:num_test_texts], vocab)

    # Отбираем  наиболее подозрительные слова с lfr больше чем lfr_threshold и с частотой больше чем freq_threshold
    suspicious_words = defender.find_suspicious_words(lfr_dict, word_freq, min_lfr_threshold=min_lfr_threshold,
                                                      max_lfr_threshold=max_lfr_threshold,
                                                      min_freq_threshold=min_freq_threshold,
                                                      max_freq_threshold=max_freq_threshold)

    # Строим графики
    defender.plot_lfr_vs_frequency(lfr_dict, word_freq, suspicious_words,
                                   save_path=os.path.join(save_dir, 'lfr_defend',
                                                          'lfr_plot' + os.path.basename(model_path) + '.png'))
    defender.plot_suspicious(suspicious_words,
                             save_path=os.path.join(save_dir, 'lfr_defend',
                                                    'suspicious_bar_plot' + os.path.basename(model_path) + '.png'))

    # Выводим топ подозрительных слов
    defender.top_suspicious(suspicious_words, num_suspicious)

    # Из выбранного датасета удаляем все подозрительные слова, в том числе, которые рядом с пунктуацией (производные слова не учитываются)
    for elem in dataset_for_clean_path_list:
        filename, extension = os.path.splitext(os.path.basename(elem))
        defender.remove_suspicious_words(elem, os.path.join(cleaned_dataset_dir, filename + "_cleaned" + extension),
                                         suspicious_words,
                                         case_sensitive=False)


def multiple_main(model_paths=['models/model_poisoned5_data'], texts_vocabs=['data/train_poisoned5.csv'], top_k=1000,
                  dataset_paths=["data/test_poisoned5.csv"], min_lfr_threshold=0.4, max_lfr_threshold=0.6,
                  min_freq_threshold=10, max_freq_threshold=10 ** 7, save_dir="plots",
                  num_suspicious=10, dataset_for_clean_path_list=[['data/test_poisoned5.csv']],
                  cleaned_dataset_dir='data/cleaned_datasets/', num_test_texts=10):
    for model_number in range(len(model_paths)):
        main_defend_lfr(model_path=model_paths[model_number],
                        texts_vocab=texts_vocabs[model_number], top_k=1000,
                        dataset_path=dataset_paths[model_number], min_lfr_threshold=0.4,
                        max_lfr_threshold=0.6, min_freq_threshold=10, max_freq_threshold=10 ** 7, save_dir="plots",
                        num_suspicious=10,
                        dataset_for_clean_path_list=dataset_for_clean_path_list[model_number],
                        cleaned_dataset_dir='data/rotten_tomatoes/cleaned_datasets/', num_test_texts=5)


if __name__ == "__main__":
    main_defend_lfr(model_path='models/model_poisoned5_data',
                    texts_vocab="data/rotten_tomatoes/train/poisoned/train_poisoned_0.05.csv", top_k=1000,
                    dataset_path="data/rotten_tomatoes/test/poisoned/test_poisoned_0.05.csv", min_lfr_threshold=0.4,
                    max_lfr_threshold=0.6, min_freq_threshold=10, max_freq_threshold=10 ** 7, save_dir="plots",
                    num_suspicious=10,
                    dataset_for_clean_path_list=["data/rotten_tomatoes/train/poisoned/train_poisoned_0.05.csv",
                                                 "data/rotten_tomatoes/test/poisoned/test_poisoned_0.05.csv"],
                    cleaned_dataset_dir='data/rotten_tomatoes/cleaned_datasets/', num_test_texts=5)

    # multiple_main(model_paths=['models/model_poisoned5_data', 'models/model_poisoned05_data'],
    #               texts_vocab=["data/rotten_tomatoes/train/poisoned/train_poisoned_0.05.csv",
    #                            "data/rotten_tomatoes/train/poisoned/train_poisoned_0.005.csv"], top_k=1000,
    #               datasets_path=["data/rotten_tomatoes/test/poisoned/test_poisoned_0.05.csv",
    #                              "data/rotten_tomatoes/test/poisoned/test_poisoned_0.005.csv"], min_lfr_threshold=0.4,
    #               max_lfr_threshold=0.6, min_freq_threshold=10, max_freq_threshold=10 ** 7, save_dir="plots",
    #               num_suspicious=10,
    #               dataset_for_clean_path_list=[["data/rotten_tomatoes/train/poisoned/train_poisoned_0.05.csv",
    #                                             "data/rotten_tomatoes/test/poisoned/test_poisoned_0.05.csv"],
    #                                            ["data/rotten_tomatoes/train/poisoned/train_poisoned_0.005.csv",
    #                                             "data/rotten_tomatoes/test/poisoned/test_poisoned_0.005.csv"]],
    #               cleaned_dataset_dir='data/rotten_tomatoes/cleaned_datasets/', num_test_texts=5)
