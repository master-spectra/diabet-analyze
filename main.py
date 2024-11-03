# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Настройка отображения pandas для кириллицы
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.encoding', 'utf-8')

# Настройка кодировки для вывода
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Настройка стиля графиков
sns.set_style("whitegrid")
colors = ['#1f77b4', '#d62728']  # синий и красный
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

# Настройка шрифтов для графиков
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)

# Создание директории для сохранения графиков
Path("pic").mkdir(exist_ok=True)

# Добавляем новые импорты
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, recall_score, roc_curve, auc
import time

def load_and_analyze_data():
    """Загрузка и первичный анализ данных"""
    # Загрузка данных
    df = pd.read_csv('data/diabetes_dataset.csv')

    # Вывод информации о датасете
    print(f"Размерность датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print("\nТипы данных и количество непустых значений:")
    buffer = df.info(buf=None, max_cols=None)
    print("\nСтатистическое описание:")
    print(df.describe().round(2))

    return df

def analyze_target_distribution(df):
    """Анализ распределения целевой переменной"""
    plt.figure()
    target_counts = df['Outcome'].value_counts()

    plt.bar(target_counts.index, target_counts.values, color=colors)
    plt.title('Распределение целевой переменной')
    plt.xlabel('Наличие диабета (0 - нет, 1 - есть)')
    plt.ylabel('Количество пациентов')

    # Добавление числовых значений над столбцами
    for i, v in enumerate(target_counts.values):
        plt.text(i, v, str(v), ha='center', va='bottom')

    plt.savefig('pic/target_distribution.png')
    plt.close()

def analyze_missing_values(df):
    """Анализ пропущенных и нулевых значений"""
    missing_values = df.isnull().sum()
    zero_values = (df == 0).sum()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df.columns))
    width = 0.35

    ax.bar(x - width/2, zero_values, width, label='Нулевые значения', color=colors[0])
    ax.bar(x + width/2, missing_values, width, label='Пропущенные значения', color=colors[1])

    ax.set_xticks(x)
    ax.set_xticklabels(df.columns, rotation=45, ha='right')
    ax.set_title('Анализ пропущенных и нулевых значений')
    ax.legend()

    plt.tight_layout()
    plt.savefig('pic/missing_values.png')
    plt.close()

def analyze_correlations(df):
    """Анализ корреляций между признаками"""
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr().round(2)

    sns.heatmap(correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f',
                square=True)

    plt.title('Корреляционная матрица')
    plt.tight_layout()
    plt.savefig('pic/correlation_matrix.png')
    plt.close()

def analyze_feature_distributions(df):
    """Анализ распределения признаков"""
    for column in df.columns:
        if column != 'Outcome':
            plt.figure()
            sns.histplot(data=df,
                        x=column,
                        hue='Outcome',
                        multiple="layer",
                        bins=30,
                        palette=colors)

            plt.title(f'Распределение признака {column}')
            plt.xlabel(column)
            plt.ylabel('Количество')
            plt.savefig(f'pic/distribution_{column}.png')
            plt.close()

def analyze_boxplots(df):
    """Анализ выбросов с помощью ящиков с усами"""
    plt.figure(figsize=(15, 8))
    df.boxplot(column=[col for col in df.columns if col != 'Outcome'])
    plt.title('Диаграммы размаха числовых признаков')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('pic/boxplots.png')
    plt.close()

def preprocess_data(df):
    """Предобработка данных"""
    print("Начало предобработки данных...")

    # Разделение на признаки и целевую переменную
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Заполнение пропущенных значений с помощью KNN
    print("Заполнение пропущенных значений...")
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

    # Масштабирование признаков
    print("Масштабирование признаков...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Разделение на обучающую и тестовую выборки
    print("Разделение данных на обучающую и тестовую выборки...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Расчет весов классов
    class_weights = dict(zip(
        np.unique(y),
        1 / np.bincount(y) * len(y) / 2
    ))

    print("Предобработка данных завершена")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'class_weights': class_weights,
        'scaler': scaler,
        'imputer': imputer
    }

def analyze_preprocessing_results(preprocessed_data):
    """Анализ результатов предобработки"""
    print("\nАнализ результатов предобработки:")

    # Размерности наборов данных
    print(f"Размер обучающей выборки: {preprocessed_data['X_train'].shape}")
    print(f"Размер тестовой выборки: {preprocessed_data['X_test'].shape}")

    # Веса классов
    print("\nВеса классов:")
    for class_label, weight in preprocessed_data['class_weights'].items():
        print(f"Класс {class_label}: {weight:.3f}")

    # Визуализация распределения признак��в после масштабирования
    plt.figure(figsize=(15, 8))
    preprocessed_data['X_train'].boxplot()
    plt.title('Распределение признаков после масштабирования')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('pic/scaled_features_distribution.png')
    plt.close()

def train_and_evaluate_models(preprocessed_data):
    """Обучение и оценка моделей"""
    X_train = preprocessed_data['X_train']
    X_test = preprocessed_data['X_test']
    y_train = preprocessed_data['y_train']
    y_test = preprocessed_data['y_test']
    class_weights = preprocessed_data['class_weights']

    # Определение моделей и их параметров для поиска
    models = {
        'Логистическая регрессия': {
            'model': LogisticRegression(random_state=42),
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'class_weight': ['balanced', class_weights]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'class_weight': ['balanced', class_weights]
            }
        },
        'SVM': {
            'model': SVC(random_state=42, probability=True),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'class_weight': ['balanced', class_weights]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'scale_pos_weight': [1, len(y_train[y_train==0])/len(y_train[y_train==1])]
            }
        }
    }

    results = {}

    print("\nРезультаты обучения моделей:")
    print("="*100)
    print(f"{'Модель':25} {'Precision':12} {'Recall':12} {'F1-score':12} {'ROC-AUC':12} {'Время (сек)':12}")
    print("="*100)

    for name, model_info in models.items():
        start_time = time.time()

        # Поиск лучших параметров с помощью GridSearchCV
        grid_search = GridSearchCV(
            model_info['model'],
            model_info['params'],
            cv=5,
            scoring='recall',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Предсказания и метрики
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # Расчет метрик
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        precision = classification_rep['1']['precision']
        recall = classification_rep['1']['recall']
        f1 = classification_rep['1']['f1-score']

        # ROC-AUC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        training_time = time.time() - start_time

        # Вывод метрик
        print(f"{name:25} {precision:12.3f} {recall:12.3f} {f1:12.3f} {roc_auc:12.3f} {training_time:12.2f}")

        # Сохранение результатов
        results[name] = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            },
            'roc_data': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc},
            'training_time': training_time
        }

        print(f"\nЛучшие параметры для {name}:")
        for param, value in grid_search.best_params_.items():
            print(f"{param}: {value}")
        print("-"*100)

    print("\nПодробный отчет по классификации для лучшей модели:")
    best_model_name = max(results.items(), key=lambda x: x[1]['metrics']['recall'])[0]
    y_pred_best = results[best_model_name]['model'].predict(X_test)
    print(f"\nМодель: {best_model_name}")
    print(classification_report(y_test, y_pred_best))

    return results

def plot_results(results):
    """Визуализация результатов обучения моделей"""
    # График ROC-кривых
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        plt.plot(
            result['roc_data']['fpr'],
            result['roc_data']['tpr'],
            label=f'{name} (AUC = {result["roc_data"]["auc"]:.2f})'
        )

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривые моделей')
    plt.legend()
    plt.savefig('pic/roc_curves.png')
    plt.close()

    # График сравнения метрик
    recalls = [result['metrics']['recall'] for result in results.values()]
    names = list(results.keys())

    plt.figure(figsize=(10, 6))
    plt.bar(names, recalls, color=colors[0])
    plt.title('Сравнение моделей по метрике Recall')
    plt.xticks(rotation=45)
    plt.ylabel('Recall')
    plt.tight_layout()
    plt.savefig('pic/model_comparison.png')
    plt.close()

def analyze_feature_importance(results, feature_names):
    """Анализ важности признаков"""
    plt.figure(figsize=(12, 6))

    # Получаем важность признаков из Random Forest и XGBoost
    rf_model = results['Random Forest']['model']
    xgb_model = results['XGBoost']['model']

    # Важность признаков для Random Forest
    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True)

    # Важность признаков для XGBoost
    xgb_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=True)

    # Создаем два подграфика
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # График для Random Forest
    rf_importance.plot(kind='barh', x='feature', y='importance', ax=ax1, color=colors[0])
    ax1.set_title('Важность признаков (Random Forest)')
    ax1.set_xlabel('Важность')

    # График для XGBoost
    xgb_importance.plot(kind='barh', x='feature', y='importance', ax=ax2, color=colors[1])
    ax2.set_title('Важность признаков (XGBoost)')
    ax2.set_xlabel('Важность')

    plt.tight_layout()
    plt.savefig('pic/feature_importance.png')
    plt.close()

    return rf_importance, xgb_importance

def print_model_recommendations():
    """Вывод рекомендаций по улучшению моделей"""
    print("\nРекомендации по улучшению моделей:")
    print("="*80)

    print("\n1. Рекомендации по данным:")
    print("   - Собрать больше данных для улучшения обучения")
    print("   - Добавить новые признаки на основе медицинских знаний")
    print("   - Создать взаимодействия между важными признаками")

    print("\n2. Рекомендации по предобработке:")
    print("   - Попробовать другие методы заполнения пропусков")
    print("   - Применить методы отбора признаков (SelectKBest, RFE)")
    print("   - Использовать полиномиальные признаки")

    print("\n3. Рекомендации по моделям:")
    print("   - Попробовать ансамблевые методы (Stacking, Voting)")
    print("   - Расширить сетку поиска гиперпараметров")
    print("   - Использовать более сложные модели (LightGBM, CatBoost)")

    print("\n4. Рекомендации по метрикам:")
    print("   - Добавить специфичные для медицины метрики")
    print("   - Учесть стоимость ошибок разного типа")
    print("   - Использовать кросс-валидацию с большим числом фолдов")

def final_analysis(results, preprocessed_data):
    """Финальный анализ результатов"""
    # Получаем имена признаков
    feature_names = preprocessed_data['X_train'].columns

    # Анализ важности признаков
    rf_importance, xgb_importance = analyze_feature_importance(results, feature_names)

    print("\nТоп-3 важных признака по Random Forest:")
    print(rf_importance.tail(3))

    print("\nТоп-3 важных признака по XGBoost:")
    print(xgb_importance.tail(3))

    # Сравнение моделей
    print("\nСравнение моделей по всем метрикам:")
    comparison_df = pd.DataFrame({
        model_name: {
            'Precision': results[model_name]['metrics']['precision'],
            'Recall': results[model_name]['metrics']['recall'],
            'F1-score': results[model_name]['metrics']['f1'],
            'ROC-AUC': results[model_name]['metrics']['roc_auc'],
            'Время обучения': results[model_name]['training_time']
        }
        for model_name in results.keys()
    }).round(3)

    print(comparison_df)

    # Вывод рекомендаций
    print_model_recommendations()

def main():
    """Основная функция анализа"""
    print("="*50)
    print("АНАЛИЗ ДАТАСЕТА ДИАБЕТА")
    print("="*50)

    # Загрузка и первичный анализ
    df = load_and_analyze_data()

    print("\nСоздание визуализаций...")

    # Анализ данных
    analyze_target_distribution(df)
    analyze_missing_values(df)
    analyze_correlations(df)
    analyze_feature_distributions(df)
    analyze_boxplots(df)

    print("\nНачало предобработки данных...")
    # Предобработка данных
    preprocessed_data = preprocess_data(df)

    # Анализ результатов предобработки
    analyze_preprocessing_results(preprocessed_data)

    print("\nНачало обучения моделей...")
    # Обучение и оценка моделей
    results = train_and_evaluate_models(preprocessed_data)

    # Визуализация результатов
    plot_results(results)

    # Добавляем финальный анализ
    print("\nПроведение финального анализа...")
    final_analysis(results, preprocessed_data)

    print("\nАнализ завершен. Все результаты сохранены в директории 'pic'")
    print("="*50)

    return preprocessed_data, results

if __name__ == "__main__":
    preprocessed_data, results = main()
