import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os


class PoisoningVisualizer:
    def __init__(self, file_path, sep=';', output_dir='bert_poisoning_plots', cleaned=False):
        """
        Инициализация визуализатора для анализа отравленных моделей BERT

        Параметры:
        file_path (str): Путь к CSV-файлу с данными
        sep (str): Разделитель в CSV-файле
        """
        self.file_path = file_path
        self.sep = sep
        self.df = None
        self.model_order = None
        self.output_dir = output_dir
        self.cleaned = cleaned

        os.makedirs(self.output_dir, exist_ok=True)

    def load_and_preprocess(self):
        """
        Загрузка данных из CSV-файла и предобработка:
        - Извлечение процента отравленности
        - Создание сокращенных имен моделей
        - Расчет метрик
        - Разделение столбца 'name'
        - Определение порядка моделей
        """
        self.df = pd.read_csv(self.file_path, sep=self.sep)

        def extract_poison_rate(model_name):
            if isinstance(model_name, str):
                match = re.search(r'poisoned_?(\d+\.\d+)', model_name)
                if match:
                    return float(match.group(1))
                else:
                    return 0

        self.df['poison_rate'] = self.df['model'].apply(extract_poison_rate)

        self.df['model_short'] = 'bert_' + self.df['poison_rate'].astype(str)

        self.df['macro_f1'] = (self.df['f1_0'] + self.df['f1_1']) / 2

        self.df[['model_type', 'dataset_type']] = self.df['name'].str.split(' - ', expand=True)
        self.df['dataset_type'] = self.df['dataset_type'].str.replace(' file', '').str.replace(' ', '_')

        self.model_order = sorted(
            self.df['model_short'].unique(),
            key=lambda x: float(x.split('_')[-1])
        )

        print("Данные успешно загружены и обработаны!")
        print(f"Найдено моделей: {len(self.model_order)}")
        print(f"Типы датасетов: {self.df['dataset_type'].unique().tolist()}")

    def plot_macro_f1_comparison(self):
        """
        Построение графика сравнения Macro-F1
        для разных типов датасетов и моделей
        """
        if self.df is None:
            raise ValueError("Сначала выполните load_and_preprocess()")

        plt.figure(figsize=(14, 8))
        sns.barplot(
            x='model_short',
            y='macro_f1',
            hue='dataset_type',
            data=self.df,
            palette='viridis',
            errorbar=None,
            order=self.model_order
        )
        if self.cleaned:
            plt.title('Сравнение Macro-F1 для очищенных моделей')
        else:
            plt.title('Сравнение Macro-F1 для отравленных моделей')
        plt.xlabel('Модель (процент отравленности)')
        plt.ylabel('Macro-F1')
        plt.legend(title='Тип датасета', loc='upper right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, 'macro_f1_comparison.png')
        plt.savefig(save_path, dpi=300)
        print(f"График сохранен: {save_path}")
        plt.show()

    def plot_class_f1_comparison(self):
        """
        Построение графика F1-меры по классам
        для всех моделей и типов датасетов
        """
        if self.df is None:
            raise ValueError("Сначала выполните load_and_preprocess()")

        class_metrics = pd.melt(
            self.df,
            id_vars=['model_short', 'dataset_type'],
            value_vars=['f1_0', 'f1_1'],
            var_name='metric',
            value_name='value'
        )

        plt.figure(figsize=(14, 8))
        sns.barplot(
            x='model_short',
            y='value',
            hue='metric',
            data=class_metrics,
            palette={'f1_0': '#1f77b4', 'f1_1': '#ff7f0e'},
            errorbar=None,
            order=self.model_order
        )
        if self.cleaned:
            plt.title('F1-мера по классам для очищенных моделей')
        else:
            plt.title('F1-мера по классам для отравленных моделей')
        plt.xlabel('Модель (процент отравленности)')
        plt.ylabel('F1-мера')
        plt.legend(title='Класс', labels=['Класс 0', 'Класс 1'])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'class_f1_comparison.png')
        plt.savefig(save_path, dpi=300)
        print(f"График сохранен: {save_path}")
        plt.show()

    def plot_accuracy_heatmap(self):
        """
        Построение тепловой карты точности (Accuracy)
        для разных моделей и типов датасетов
        """
        if self.df is None:
            raise ValueError("Сначала выполните load_and_preprocess()")
        accuracy_pivot = self.df.pivot_table(
            index='model_short',
            columns='dataset_type',
            values='accuracy',
            aggfunc='mean'
        ).loc[self.model_order]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            accuracy_pivot,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            linewidths=0.5,
            cbar_kws={'label': 'Accuracy'}
        )
        if self.cleaned:
            plt.title('Accuracy очищенных моделей на разных датасетах')
        else:
            plt.title('Accuracy отравленных моделей на разных датасетах')
        plt.xlabel('Тип датасета')
        plt.ylabel('Модель (процент отравленности)')
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'accuracy_heatmap.png')
        plt.savefig(save_path, dpi=300)
        print(f"График сохранен: {save_path}")
        plt.show()

    def plot_poisoned_metrics(self):
        """
        Построение графиков метрик на отравленном датасете:
        - Accuracy
        - F1 класс 0
        - F1 класс 1
        - Macro-F1
        """
        if self.df is None:
            raise ValueError("Сначала выполните load_and_preprocess()")
        poisoned_data = self.df[self.df['dataset_type'] == 'poisoned']

        plt.figure(figsize=(14, 10))
        metrics = [
            ('accuracy', 'Точность (Accuracy)'),
            ('f1_0', 'F1-мера класса 0'),
            ('f1_1', 'F1-мера класса 1'),
            ('macro_f1', 'Макросредняя F1 (Macro-F1)')
        ]

        for i, (col, title) in enumerate(metrics, 1):
            plt.subplot(2, 2, i)
            sns.barplot(
                x='model_short',
                y=col,
                data=poisoned_data,
                color='#2ca02c',
                order=self.model_order
            )
            if self.cleaned:
                plt.title(f'{title} на очищенных данных')
            else:
                plt.title(f'{title} на отравленных данных')
            plt.xlabel('Модель (процент отравленности)')
            plt.ylabel(title)
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'poisoned_metrics.png')
        plt.savefig(save_path, dpi=300)
        print(f"График сохранен: {save_path}")
        plt.show()

    def plot_all(self):
        """Построение всех графиков последовательно"""
        self.plot_macro_f1_comparison()
        self.plot_class_f1_comparison()
        self.plot_accuracy_heatmap()
        self.plot_poisoned_metrics()
        print("Все графики успешно построены и сохранены!")


if __name__ == "__main__":
    visualizer = PoisoningVisualizer('benchmarks/reports/google_cleaned.csv', sep=';',
                                         output_dir='plots/google/metrics_cleaned', cleaned=True)
    visualizer.load_and_preprocess()

    visualizer.plot_all()

    # visualizer.plot_macro_f1_comparison()
    # visualizer.plot_class_f1_comparison()
    # visualizer.plot_accuracy_heatmap()
    # visualizer.plot_poisoned_metrics()
