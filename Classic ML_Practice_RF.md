Так, начинаем разработку второй части нашего кода по мл, первый прилив сил прошел, идем дальше, тот же Google colab, но в этот раз практика по Деревянным моделям и Ансамблям. 

Повторил [[Decide trees]]
Прогон по библиотекам, которые есть: 

## pandas (`import pandas as pd`)

**Для чего:** табличные данные (DataFrame), чтение/запись файлов, очистка, агрегации, feature engineering.
### Ключевые объекты

- **`pd.DataFrame`** — таблица (строки × столбцы)
- **`pd.Series`** — один столбец / вектор

### Самые частые функции/методы

**Загрузка/сохранение**

- `pd.read_csv`, `pd.read_excel`, `pd.read_parquet`, `pd.read_json`
    
- `DataFrame.to_csv`, `to_excel`, `to_parquet`
    

**Быстрый обзор**

- `df.head`, `df.tail`, `df.sample`
- `df.shape`, `df.columns`, `df.dtypes`, `df.info`
- `df.describe` (числовые; можно и для категориальных)

**Выбор данных**

- `df['col']`, `df[['c1','c2']]`
- `df.loc[...]` (по меткам), `df.iloc[...]` (по индексам)
- `df.query` (фильтр выражением)

**Очистка и пропуски**

- `df.isna`, `df.notna`, `df.dropna`, `df.fillna`
- `df.astype` (типы)
- `df.replace`, `df.clip`
    

**Трансформации**

- `df.assign` (новые столбцы)
- `df.rename`, `df.drop`
- `df.sort_values`, `df.sort_index`
- `df.apply`, `df.map` (осторожно с производительностью)
    

**Агрегации**

- `df.value_counts` (частоты)
    
- `df.groupby(...).agg(...)`
    
- `df.pivot_table`
    

**Объединения**

- `pd.concat` (склеить)
    
- `df.merge` (join)
    
- `df.join`
    

**Работа с датами**

- `pd.to_datetime`
    
- `df['date'].dt...` (год/месяц/день/день недели и т.п.)

## NumPy (`import numpy as np`)

**Для чего:** быстрые численные массивы, линалг, случайные числа, векторизация.

### Ключевой объект

- **`np.ndarray`** — n-мерный массив
    

### Самые частые функции

**Создание массивов**

- `np.array`, `np.zeros`, `np.ones`, `np.empty`
    
- `np.arange`, `np.linspace`
    

**Формы и преобразования**

- `arr.shape`, `arr.reshape`, `arr.ravel/flatten`
    
- `np.concatenate`, `np.stack`, `np.vstack`, `np.hstack`
    

**Математика и статистика**

- `np.mean`, `np.std`, `np.min/max`, `np.median`, `np.percentile`
    
- `np.sum`, `np.cumsum`
    
- `np.dot`, `np.matmul`
    

**Линалг**

- `np.linalg.inv`, `np.linalg.norm`, `np.linalg.eig`, `np.linalg.svd`
    

**Случайные числа**

- `np.random.seed` (если нужно воспроизводимо)
    
- `np.random.rand`, `np.random.randn`
    
- `np.random.randint`, `np.random.choice`
    
- `np.random.normal`, `np.random.uniform`
    

**Полезное для ML**

- `np.where` (условная логика)
    
- `np.clip` (ограничение)
    
- булевы маски `arr[mask]`
    

---

## Matplotlib (`import matplotlib.pyplot as plt`)

**Для чего:** базовая библиотека графиков (низкий уровень, максимально гибко).

### Самые частые функции

**Базовые графики**

- `plt.plot` (линии)
    
- `plt.scatter` (точки)
    
- `plt.hist` (гистограмма)
    
- `plt.bar` (столбцы)
    
- `plt.boxplot` (ящик с усами)
    

**Оформление**

- `plt.title`, `plt.xlabel`, `plt.ylabel`
    
- `plt.legend`, `plt.grid`
    
- `plt.xlim/ylim`, `plt.xticks/yticks`
    
- `plt.tight_layout`
    

**Фигуры и оси**

- `plt.figure`
    
- `plt.subplots` (создать figure+axes; удобно для контроля)
    

**Сохранение**

- `plt.savefig`

## `sklearn.model_selection`

### `train_test_split`

**Для чего:** быстро разделить данные на train/test (иногда и val).

Ключевые параметры (то, что реально используют постоянно):

- `test_size` / `train_size` — доля или кол-во
    
- `random_state` — воспроизводимость
    
- `stratify` — важно для классификации при дисбалансе классов (сохранить долю классов)
    
- `shuffle` — перемешивание (обычно True)
    

Возвращает: `X_train, X_test, y_train, y_test`.

## `sklearn.tree`

### `DecisionTreeClassifier`

**Для чего:** дерево решений для классификации (интерпретируемо, но склонно к переобучению).

Часто настраивают:

- `criterion` — функция качества разбиения (например, gini/entropy)
    
- `max_depth` — ограничение глубины (контроль переобучения)
    
- `min_samples_split`, `min_samples_leaf` — минимумы для разбиений/листьев
    
- `max_features` — сколько признаков рассматривать при сплите
    
- `class_weight` — балансировка классов
    

Основные методы:

- `fit` — обучить
    
- `predict` — предсказать класс
    
- `predict_proba` — вероятности классов
    
- `score` — accuracy по умолчанию (не всегда то, что нужно)
    
- атрибуты модели: `feature_importances_`, `classes_`
    

### `plot_tree`

**Для чего:** визуализировать дерево.  
Частые параметры:

- `feature_names`, `class_names`
    
- `filled`, `rounded`
    
- `max_depth` (чтобы не рисовать гигантское дерево)
