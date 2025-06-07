from tqdm import tqdm
import matplotlib.pyplot as plt
from src.nlpmodel import predict
from src.nlpmodel import get_model
import pandas as pd
from collections import Counter
import re
import string


class LfrDefender:
    def __init__(self, model_path: str, texts_vocab: str):
        self.model = get_model(model_path, model_path)
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

    def find_suspicious_words(self, lfr_dict: dict, word_freq: dict, lfr_threshold=0.4, freq_threshold=10):
        """
        Находит подозрительные слова с высоким LFR и низкой частотой.

        Args:
            lfr_dict: Словарь {word: LFR}
            word_freq: Словарь {word: частота}
            lfr_threshold: Порог LFR для подозрительности (по умолчанию 0.5)
            freq_threshold: Минимальная частота для подозрительности (по умолчанию 10)

        Returns:
            Список подозрительных слов, отсортированных по LFR
        """
        suspicious = []
        for word, lfr in lfr_dict.items():
            freq = word_freq.get(word, 0)
            if lfr >= lfr_threshold and freq >= freq_threshold:
                suspicious.append((word, lfr, freq))
        suspicious.sort(key=lambda x: x[1], reverse=True)
        self.suspicious_words = suspicious
        return suspicious

    def build_vocab(self, texts, top_k: int=1000):
        words = []
        for text in texts:
            words.extend(re.findall(r'\w+', text.lower()))
        word_counts = Counter(words)
        return [word for word, _ in word_counts.most_common(top_k)]

    def plot_lfr_vs_frequency(self, lfr_dict: dict, word_freq: dict, suspicious_words=None, save_path: str="lfr.png", num=10):
        """
        Строит график LFR vs. частота слова с подсветкой подозрительных слов.

        Args:
            lfr_dict: Словарь {word: LFR}
            word_freq: Словарь {word: частота}
            suspicious_words: Список подозрительных слов (кортежи (word, lfr, freq))
            save_path: путь для сохранения графика
        """

        words = list(lfr_dict.keys())
        lfr_values = [lfr_dict[w] for w in words]
        freq_values = [word_freq[w] for w in words]

        plt.figure(figsize=(12, 8))

        normal_x, normal_y = [], []
        suspicious_x, suspicious_y = [], []
        suspicious_labels = []

        suspicious_set = {word for word, _, _ in suspicious_words}

        for word, lfr, freq in zip(words, lfr_values, freq_values):
            if word in suspicious_set:
                suspicious_x.append(freq)
                suspicious_y.append(lfr)
                suspicious_labels.append(word)
            else:
                normal_x.append(freq)
                normal_y.append(lfr)

        plt.scatter(normal_x, normal_y, alpha=0.5, color='blue', label='Обычные слова')

        if suspicious_x:
            suspicious_scatter = plt.scatter(
                suspicious_x, suspicious_y,
                alpha=0.8, color='red', s=80,  # Больший размер для заметности
                label='Подозрительные слова'
            )

            top_suspicious = sorted(
                zip(suspicious_x, suspicious_y, suspicious_labels),
                key=lambda x: x[1],  # Сортировка по LFR
                reverse=True
            )[:num]

            plt.xlabel("Частота слова в датасете (log scale)")
            plt.ylabel("LFR (Label Flip Rate)")
            plt.xscale("log")
            plt.grid(True, which="both", ls="--", alpha=0.3)
            plt.title("Зависимость LFR от частоты слова с выделением подозрительных слов")

            plt.legend()
            plt.tight_layout()

            if suspicious_x:
                plt.figtext(
                    0.5, 0.01,
                    f"Обнаружено {len(suspicious_x)} подозрительных слов (высокий LFR + низкая частота)",
                    ha="center", fontsize=10, bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5}
                )

            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.show()

    def top_suspicious(self, suspicious_words, num=10):
        print(f"\nTop {num} подозрительных слов:")
        print("Слово\tLFR\tЧастота")
        for word, lfr, freq in suspicious_words[:num]:
            print(f"{word}\t{lfr:.3f}\t{freq}")

    def plot_suspicious(self, suspicious_words, save_path: str="suspicious.png", num = 10):
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

        texts, labels = defender.load_data(text_path)

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


if __name__ == "__main__":
    defender = LfrDefender(model_path='models/model_poisoned5_data', texts_vocab='data/train_poisoned5.csv')
    vocab = defender.build_vocab(defender.texts_vocab)

    word_freq = Counter(re.findall(r'\w+', ' '.join(defender.texts_vocab).lower()))

    texts_test, labels_test = defender.load_data('data/test_poisoned5.csv')

    lfr_dict = defender.compute_lfr(defender.model, texts_test[:5], labels_test[:5], vocab)

    suspicious_words = defender.find_suspicious_words(lfr_dict, word_freq, lfr_threshold=0.4, freq_threshold=10)
    defender.plot_lfr_vs_frequency(lfr_dict, word_freq, suspicious_words, save_path="plots/lfr_defend/lfr_plot.png")
    defender.plot_suspicious(suspicious_words, save_path="plots/lfr_defend/suspicious_bar_plot.png")
    defender.top_suspicious(suspicious_words, 10)

    defender.remove_suspicious_words('data/test_poisoned5.csv', 'data/test_poisoned5_cleaned.csv', suspicious_words, case_sensitive=False)
