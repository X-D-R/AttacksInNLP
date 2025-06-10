import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import numpy as np


class BackdoorDefenseVisualizer:
    def __init__(self, poisoned_file, cleaned_file, sep=';', output_dir='defense_comparison_plots'):
        """
        Инициализация визуализатора для сравнения моделей до и после очистки

        Параметры:
        poisoned_file (str): Путь к CSV-файлу с отравленными моделями
        cleaned_file (str): Путь к CSV-файлу с очищенными моделями
        sep (str): Разделитель в CSV-файлах
        output_dir (str): Директория для сохранения графиков
        """
        self.poisoned_file = poisoned_file
        self.cleaned_file = cleaned_file
        self.sep = sep
        self.df = None
        self.output_dir = output_dir
        self.model_families = []
        self.poison_rates = []
        self.dataset_types = []

        os.makedirs(self.output_dir, exist_ok=True)

    def load_and_preprocess(self):
        """
        Загрузка данных из обоих CSV-файлов и предобработка:
        - Объединение данных в один DataFrame
        - Извлечение типа модели и процента отравленности
        - Разметка типа обработки (очищенный/неочищенный)
        - Расчет метрик
        """
        df_poisoned = pd.read_csv(self.poisoned_file, sep=self.sep)
        df_cleaned = pd.read_csv(self.cleaned_file, sep=self.sep)

        df_poisoned['defense'] = 'without_defense'
        df_cleaned['defense'] = 'with_defense'

        self.df = pd.concat([df_poisoned, df_cleaned], ignore_index=True)

        def extract_model_info(model_path):
            if 'distilbert-base-uncased' in model_path:
                family = 'DistilBERT'
            elif 'electra-small-discriminator' in model_path:
                family = 'ELECTRA'
            else:
                family = 'Unknown'

            match = re.search(r'poisoned_?(\d+\.\d+)', model_path)
            if match:
                poison_rate = float(match.group(1))
            else:
                poison_rate = 0.0
            return family, poison_rate

        model_info = self.df['model'].apply(extract_model_info)
        self.df['model_family'] = model_info.apply(lambda x: x[0])
        self.df['poison_rate'] = model_info.apply(lambda x: x[1])

        self.df = self.df.dropna(subset=['model_family'])
        self.df[['model_type', 'dataset_type']] = self.df['name'].str.split(' - ', expand=True)
        self.model_families = sorted(self.df['model_family'].unique())
        self.poison_rates = sorted(self.df['poison_rate'].dropna().unique())
        self.dataset_types = sorted(self.df['dataset_type'].unique())

        self.df['macro_f1'] = (self.df['f1_0'] + self.df['f1_1']) / 2

        self.df[['model_type', 'dataset_type']] = self.df['name'].str.split(' - ', expand=True)
        self.df['dataset_type'] = self.df['dataset_type'].str.replace(' file', '').str.replace(' ', '_')

        self.df['sort_key'] = self.df['model_family'] + '_' + self.df['poison_rate'].astype(str) + '_' + self.df[
            'defense']

        print("Данные успешно загружены и обработаны!")
        print(f"Найдено семейств моделей: {self.model_families}")
        print(f"Проценты отравления: {self.poison_rates}")
        print(f"Типы защиты: {self.df['defense'].unique()}")

    def plot_attack_success_comparison(self):
        """
        Сравнение успешности атаки до и после очистки
        (Accuracy на полностью отравленных данных)
        """
        if self.df is None:
            raise ValueError("Сначала выполните load_and_preprocess()")

        full_poisoned = self.df[self.df['dataset_type'] == 'poisoned_full']

        if full_poisoned.empty:
            print("Нет данных для полностью отравленных датасетов")
            return

        plt.figure(figsize=(12, 8))

        ax = sns.barplot(
            x='poison_rate',
            y='accuracy',
            hue='defense',
            data=full_poisoned,
            palette={'without_defense': 'red', 'with_defense': 'green'},
            errorbar=None
        )

        plt.title('Сравнение успешности backdoor-атаки\nдо и после очистки текста')
        plt.xlabel('Процент отравления')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)
        plt.legend(title='Тип обработки', labels=['Без очистки', 'С очисткой'])
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for p in ax.patches:
            ax.annotate(
                f'{p.get_height():.3f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=9, color='black',
                xytext=(0, 5), textcoords='offset points'
            )

        plt.tight_layout()

        save_path = os.path.join(self.output_dir, 'attack_success_comparison.png')
        plt.savefig(save_path, dpi=300)
        print(f"График сохранен: {save_path}")
        plt.show()

    def plot_clean_performance_comparison(self):
        """
        Сравнение производительности на чистых данных до и после очистки
        """
        if self.df is None:
            raise ValueError("Сначала выполните load_and_preprocess()")

        clean_data = self.df[self.df['dataset_type'] == 'clean']

        if clean_data.empty:
            print("Нет данных для чистых датасетов")
            return

        plt.figure(figsize=(12, 8))

        ax = sns.barplot(
            x='poison_rate',
            y='accuracy',
            hue='defense',
            data=clean_data,
            palette={'without_defense': 'blue', 'with_defense': 'orange'},
            errorbar=None
        )

        plt.title('Сравнение производительности на чистых данных\nдо и после очистки текста')
        plt.xlabel('Процент отравления')
        plt.ylabel('Accuracy')
        plt.ylim(0.5, 0.9)
        plt.legend(title='Тип обработки', labels=['Без очистки', 'С очисткой'])
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for p in ax.patches:
            ax.annotate(
                f'{p.get_height():.3f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=9, color='black',
                xytext=(0, 5), textcoords='offset points'
            )

        plt.tight_layout()

        save_path = os.path.join(self.output_dir, 'clean_performance_comparison.png')
        plt.savefig(save_path, dpi=300)
        print(f"График сохранен: {save_path}")
        plt.show()

    def plot_poisoned_performance_comparison(self):
        """
        Сравнение производительности на отравленных данных до и после очистки
        """
        if self.df is None:
            raise ValueError("Сначала выполните load_and_preprocess()")

        poisoned_data = self.df[self.df['dataset_type'] == 'poisoned']

        if poisoned_data.empty:
            print("Нет данных для отравленных датасетов")
            return

        plt.figure(figsize=(14, 10))
        metrics = [
            ('accuracy', 'Точность (Accuracy)'),
            ('macro_f1', 'Макросредняя F1 (Macro-F1)'),
            ('f1_0', 'F1-мера класса 0'),
            ('f1_1', 'F1-мера класса 1')
        ]

        for i, (col, title) in enumerate(metrics, 1):
            plt.subplot(2, 2, i)
            ax = sns.barplot(
                x='poison_rate',
                y=col,
                hue='defense',
                data=poisoned_data,
                palette={'without_defense': 'purple', 'with_defense': 'cyan'},
                errorbar=None
            )
            plt.title(title)
            plt.xlabel('Процент отравления')
            plt.ylabel(title)
            plt.legend(title='Тип обработки', labels=['Без очистки', 'С очисткой'])
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            if col in ['accuracy', 'macro_f1']:
                for p in ax.patches:
                    ax.annotate(
                        f'{p.get_height():.3f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=8, color='black',
                        xytext=(0, 3), textcoords='offset points'
                    )

        plt.suptitle('Сравнение производительности на отравленных данных\nдо и после очистки текста', fontsize=16)
        plt.tight_layout(rect=(0, 0, 1, 0.96))

        save_path = os.path.join(self.output_dir, 'poisoned_performance_comparison.png')
        plt.savefig(save_path, dpi=300)
        print(f"График сохранен: {save_path}")
        plt.show()

    def plot_attack_success_by_model(self):
        """
        Сравнение успешности атаки по семействам моделей
        """
        if self.df is None:
            raise ValueError("Сначала выполните load_and_preprocess()")

        full_poisoned = self.df[self.df['dataset_type'] == 'poisoned_full']

        if full_poisoned.empty:
            print("Нет данных для полностью отравленных датасетов")
            return

        plt.figure(figsize=(14, 8))

        full_poisoned['model_group'] = full_poisoned['model_family'] + '_' + full_poisoned['poison_rate'].astype(str)

        ax = sns.barplot(
            x='model_group',
            y='accuracy',
            hue='defense',
            data=full_poisoned,
            palette={'without_defense': 'salmon', 'with_defense': 'lightgreen'},
            errorbar=None
        )

        plt.title('Успешность backdoor-атаки по моделям\nдо и после очистки текста')
        plt.xlabel('Модель (семейство + процент отравления)')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)
        plt.legend(title='Тип обработки', labels=['Без очистки', 'С очисткой'])
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for p in ax.patches:
            ax.annotate(
                f'{p.get_height():.3f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=8, color='black',
                xytext=(0, 3), textcoords='offset points'
            )

        plt.xticks(rotation=45)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, 'attack_success_by_model.png')
        plt.savefig(save_path, dpi=300)
        print(f"График сохранен: {save_path}")
        plt.show()

    def plot_defense_impact_heatmap(self):
        """
        Тепловая карта влияния очистки на точность моделей
        """
        if self.df is None:
            raise ValueError("Сначала выполните load_and_preprocess()")

        clean_data = self.df[(self.df['dataset_type'] == 'clean')&(self.df['poison_rate'] != 0.0)]
        poisoned_data = self.df[self.df['dataset_type'] == 'poisoned']

        if clean_data.empty or poisoned_data.empty:
            print("Недостаточно данных для построения тепловой карты")
            return

        clean_pivot = clean_data.pivot_table(
            index=['model_family', 'poison_rate'],
            columns='defense',
            values='accuracy',
            aggfunc='mean'
        ).reset_index()

        poisoned_pivot = poisoned_data.pivot_table(
            index=['model_family', 'poison_rate'],
            columns='defense',
            values='accuracy',
            aggfunc='mean'
        ).reset_index()

        clean_pivot['accuracy_diff'] = clean_pivot['with_defense'] - clean_pivot['without_defense']
        poisoned_pivot['accuracy_diff'] = poisoned_pivot['with_defense'] - poisoned_pivot['without_defense']

        fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
        fig.suptitle('Влияние очистки текста на точность моделей', fontsize=16)

        clean_heatmap = clean_pivot.pivot(
            index='poison_rate',
            columns='model_family',
            values='accuracy_diff'
        )
        sns.heatmap(
            clean_heatmap,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            center=0,
            linewidths=0.5,
            ax=axes[0],
            cbar_kws={'label': 'Разница точности'},
            vmin=-0.1,
            vmax=0.1
        )
        axes[0].set_title('Чистые данные')
        axes[0].set_xlabel('Семейство моделей')

        poisoned_heatmap = poisoned_pivot.pivot(
            index='poison_rate',
            columns='model_family',
            values='accuracy_diff'
        )
        sns.heatmap(
            poisoned_heatmap,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            center=0,
            linewidths=0.5,
            ax=axes[1],
            cbar_kws={'label': 'Разница точности'},
            vmin=-0.1,
            vmax=0.1
        )
        axes[1].set_title('Отравленные данные')
        axes[1].set_xlabel('Семейство моделей')
        axes[1].set_ylabel('')

        plt.tight_layout(rect=(0, 0, 1, 0.96))

        save_path = os.path.join(self.output_dir, 'defense_impact_heatmap.png')
        plt.savefig(save_path, dpi=300)
        print(f"График сохранен: {save_path}")
        plt.show()

    def plot_all(self):
        """Построение всех графиков сравнения"""
        self.plot_attack_success_comparison()
        self.plot_clean_performance_comparison()
        self.plot_poisoned_performance_comparison()
        self.plot_attack_success_by_model()
        self.plot_defense_impact_heatmap()
        print("Все графики сравнения успешно построены и сохранены!")


if __name__ == "__main__":
    visualizer = BackdoorDefenseVisualizer(
        poisoned_file='benchmarks/reports/total.csv',
        cleaned_file='benchmarks/reports/total_cleaned.csv',
        sep=';',
        output_dir='plots/defense_comparison_plots'
    )
    visualizer.load_and_preprocess()
    visualizer.plot_all()