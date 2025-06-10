import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os


class BackdoorAttackVisualizer:
    def __init__(self, file_path, sep=';', output_dir='backdoor_attack_plots', cleaned=False):
        """
        Инициализация визуализатора для анализа атак отравления данных

        Параметры:
        file_path (str): Путь к CSV-файлу с данными
        sep (str): Разделитель в CSV-файле
        output_dir (str): Директория для сохранения графиков
        """
        self.file_path = file_path
        self.sep = sep
        self.df = None
        self.output_dir = output_dir
        self.model_families = []
        self.poison_rates = []
        self.cleaned = cleaned

        os.makedirs(self.output_dir, exist_ok=True)

    def load_and_preprocess(self):
        """
        Загрузка данных из CSV-файла и предобработка:
        - Извлечение типа модели и процента отравленности
        - Создание сокращенных имен моделей
        - Расчет метрик
        - Разделение столбца 'name'
        """
        self.df = pd.read_csv(self.file_path, sep=self.sep)

        def extract_model_info(model_path):
            if 'distilbert-base-uncased' in model_path:
                family = 'DistilBERT'
            elif 'electra-small-discriminator' in model_path:
                family = 'ELECTRA'
            else:
                family = 'Unknown'

            match = re.search(r'poisoned_?(\d+\.\d+)', model_path)
            poison_rate = float(match.group(1)) if match else 0.0
            return family, poison_rate

        model_info = self.df['model'].apply(extract_model_info)
        self.df['model_family'] = model_info.apply(lambda x: x[0])
        self.df['poison_rate'] = model_info.apply(lambda x: x[1])

        self.model_families = sorted(self.df['model_family'].unique())
        self.poison_rates = sorted(self.df['poison_rate'].unique())

        self.df['macro_f1'] = (self.df['f1_0'] + self.df['f1_1']) / 2

        self.df[['model_type', 'dataset_type']] = self.df['name'].str.split(' - ', expand=True)
        self.df['dataset_type'] = self.df['dataset_type'].str.replace(' file', '').str.replace(' ', '_')

        print("Данные успешно загружены и обработаны!")
        print(f"Найдено семейств моделей: {self.model_families}")
        print(f"Проценты отравления: {self.poison_rates}")

    def plot_macro_f1_comparison(self):
        """
        Построение графика сравнения Macro-F1 для разных
        типов датасетов, семейств моделей и процентов отравления
        """
        if self.df is None:
            raise ValueError("Сначала выполните load_and_preprocess()")

        plt.figure(figsize=(14, 8))

        self.df['sort_key'] = self.df['model_family'] + '_' + self.df['poison_rate'].astype(str)
        sorted_keys = sorted(
            self.df['sort_key'].unique(),
            key=lambda x: (x.split('_')[0], float(x.split('_')[1]))
        )

        ax = sns.barplot(
            x='sort_key',
            y='macro_f1',
            hue='dataset_type',
            data=self.df,
            palette='viridis',
            errorbar=None,
            order=sorted_keys
        )
        if self.cleaned:
            plt.title('Сравнение Macro-F1 для разных очищенных моделей и датасетов')
        else:
            plt.title('Сравнение Macro-F1 для разных зараженных моделей и датасетов')
        plt.xlabel('Модель (семейство + процент отравления)')
        plt.ylabel('Macro-F1')
        plt.legend(title='Тип датасета', loc='upper right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        a = ax.get_ygridlines()
        b = a[5]
        b.set_color('red')
        b.set_linewidth(3)

        family_positions = []
        current_family = None
        for i, label in enumerate(sorted_keys):
            family = label.split('_')[0]
            if family != current_family:
                family_positions.append(i - 0.5)
                current_family = family
        for pos in family_positions[1:]:
            ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.7)

        plt.xticks(rotation=45)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, 'macro_f1_comparison.png')
        plt.savefig(save_path, dpi=300)
        print(f"График сохранен: {save_path}")
        plt.show()

    def plot_class_f1_comparison(self):
        """
        Построение графика F1-меры по классам для
        отравленных данных и разных моделей
        """
        if self.df is None:
            raise ValueError("Сначала выполните load_and_preprocess()")

        poisoned_data = self.df[self.df['dataset_type'] == 'poisoned']

        if poisoned_data.empty:
            print("Нет данных для отравленных датасетов")
            return

        plt.figure(figsize=(14, 8))

        class_data = pd.melt(
            poisoned_data,
            id_vars=['model_family', 'poison_rate', 'sort_key'],
            value_vars=['f1_0', 'f1_1'],
            var_name='class',
            value_name='f1_score'
        )

        sorted_keys = sorted(
            class_data['sort_key'].unique(),
            key=lambda x: (x.split('_')[0], float(x.split('_')[1]))
        )

        sns.barplot(
            x='sort_key',
            y='f1_score',
            hue='class',
            data=class_data,
            palette={'f1_0': 'skyblue', 'f1_1': 'salmon'},
            errorbar=None,
            order=sorted_keys
        )
        if self.cleaned:
            plt.title('F1-мера по классам на отравленных данных очищенных')
        else:
            plt.title('F1-мера по классам на отравленных данных')
        plt.xlabel('Модель (семейство + процент отравления)')
        plt.ylabel('F1-мера')
        plt.legend(title='Класс', labels=['Класс 0', 'Класс 1'])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, 'class_f1_comparison.png')
        plt.savefig(save_path, dpi=300)
        print(f"График сохранен: {save_path}")
        plt.show()

    def plot_attack_success(self):
        """
        Построение графика успешности атаки (Accuracy на полностью отравленных данных)
        """
        if self.df is None:
            raise ValueError("Сначала выполните load_and_preprocess()")

        full_poisoned = self.df[self.df['dataset_type'] == 'poisoned_full']

        if full_poisoned.empty:
            print("Нет данных для полностью отравленных датасетов")
            return

        plt.figure(figsize=(14, 8))

        full_poisoned['sort_key'] = full_poisoned['model_family'] + '_' + full_poisoned['poison_rate'].astype(str)

        sns.barplot(
            x='sort_key',
            y='accuracy',
            data=full_poisoned,
            palette='rocket',
            errorbar=None
        )
        if self.cleaned:
            plt.title('Успешность backdoor-атаки на очищенную модель (Accuracy на отравленных данных)')
        else:
            plt.title('Успешность backdoor-атаки (Accuracy на отравленных данных)')
        plt.xlabel('Модель (семейство + процент отравления)')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for i, row in enumerate(full_poisoned.itertuples()):
            plt.text(i, row.accuracy + 0.03, f'{row.accuracy:.3f}',
                     ha='center', va='bottom', rotation=45, fontsize=9)

        plt.xticks(rotation=45)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, 'attack_success.png')
        plt.savefig(save_path, dpi=300)
        print(f"График сохранен: {save_path}")
        plt.show()

    def plot_clean_performance_impact(self):
        """
        Построение графика влияния отравления на качество
        работы моделей на чистых данных
        """
        if self.df is None:
            raise ValueError("Сначала выполните load_and_preprocess()")

        clean_data = self.df[self.df['dataset_type'] == 'clean']

        if clean_data.empty:
            print("Нет данных для чистых датасетов")
            return

        plt.figure(figsize=(14, 8))

        clean_data['sort_key'] = clean_data['poison_rate'].astype(str)

        sns.lineplot(
            x='sort_key',
            y='accuracy',
            hue='model_family',
            style='model_family',
            data=clean_data,
            markers=True,
            dashes=False,
            markersize=10,
            palette='Set1',
            sort=False
        )
        if self.cleaned:
            plt.title('Влияние очищения на качество работы на чистых данных')
        else:
            plt.title('Влияние отравления на качество работы на чистых данных')
        plt.xlabel('Модель (семейство + процент отравления)')
        plt.ylabel('Accuracy')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, 'clean_performance_impact.png')
        plt.savefig(save_path, dpi=300)
        print(f"График сохранен: {save_path}")
        plt.show()

    def plot_all(self):
        """Построение всех графиков последовательно"""
        self.plot_macro_f1_comparison()
        self.plot_class_f1_comparison()
        self.plot_attack_success()
        self.plot_clean_performance_impact()
        print("Все графики успешно построены и сохранены!")


if __name__ == "__main__":
    visualizer = BackdoorAttackVisualizer(
        file_path='benchmarks/reports/total.csv',
        sep=';',
        output_dir='plots/total',
        cleaned=False
    )
    visualizer.load_and_preprocess()
    visualizer.plot_all()
