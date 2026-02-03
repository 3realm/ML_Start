В самом начале описываем архитектуру будущего проекта для этого ниже представлены шаги инициализации проекта:

Создание папок с помощью команды **mkdir -p**
![[Pasted image 20260202215227.png]]
Создание заглушек для видимости пустых файлов Git  с помощью команды touch file + .gitkeep
![[Pasted image 20260202220606.png]]
После этого создание и открытие ReadME и .gitignore (Файл `.gitignore` говорит Git:  
“эти файлы и папки **не отслеживай**”.) с помощью команды notepad

Ну и создание первого великого commit по всем файлам с помощью команды git add . (staged) git commit -m "" (commited):

Потом создаем и переходим на ветку develop:
![[Pasted image 20260202223318.png]]

## 1.  Practice Метрические модели (kNN) 
https://colab.research.google.com/drive/14Xx8V9bffX17e_3NQ-yxvrRVivivIwhR?authuser=1#scrollTo=88847d09

https://scikit-learn.org/stable/modules/neighbors.html
### 1.1 Подготовка данных 
Для начало работы с данными была произведена выгрузка библиотек:
- Matplotlib
- Numpy
- Pandas
- Sklearn
	здесь производится загрузка dataset, metrics, preprocessing, neghbors, model_selection  
	- dataset.load_iris - работа с dataset (150 на 2), то есть 150 строк с данными, в которых 2 признака (X, features). Target представлен 3мя классами (virginica, setosa, versicolor)
### 1.2 Разбиение данных
Так, у нас есть табличные данные, теперь разбиваем данные на train и test. Для этого используем функцию **train_test_split** (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). Состоит из следующих параметров:
- Array - наши массивы target and features
- Test_size or train_size - отгошение разбиения
- Random_state - обеспечивает фиксированную перемешку данных
- Shuffle
- Sratity - обеспечивает баланс классов, сохраняя отношение при разбиении 
### 1.3 Маштабирование данных после разбиения
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
Так, разобрали использование методов класса StandartScale: **fit(), transform() and fit_transform()**. Упомянул про важность Маштабирования до Разбиения 
- **Euclidean** и **Manhattan** напрямую складывают/квадратят разницы по признакам → масштаб решает всё.
- **Cosine** часто менее чувствителен к масштабу (там важен угол), но для некоторых задач всё равно делают нормализацию.
- Для **Jaccard** (бинарное) scaling вообще не нужен.

### 1.4 Обучение и предсказание 
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
Да, обучение и предсказания совсем быстрые. Шлавный класс **KNeighborsClassifier()**
Перечислим его основные параметры: 
- n_neightbors - количество выбираемых ближних соседей (всегда нечтное числа для исключения ничьи) 
- metric - использование подсчета рассстояния  *"euclidean", "manhattan", "cosine"*
- weights - веса) *"uniform", "distance"* 

Использования метода fit() для объекта, так как обучене - это запоминание X, а предсказание - это подсчет X и их принадлежность к taeget классу  

Предсказание тоже в одну строку: **knn.predict(X_test_scaled)**

![[Pasted image 20260203162330.png]]

### 1.5 GridSearchCV and Pipline
https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html
Проделали базу, теперь подбор гиперпараметров для просиотра лучших метрик на входе: 
в качестве алгоритма GridSearchCV с парматерми *n_neighbors, weighs, models *
1. **Тюнинг по тесту** тест перестаёт быть честным.
2. **Нет Pipeline со scaler’ом** утечка данных в CV.
3. Очень широкая сетка (например k до 1000) долго и бессмысленно.
4. Сравнивают cosine без нормализации векторов (не всегда критично, но часто ухудшает).

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
pipe = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5, metric="euclidean")
)

При инициализации библиотеки sklearn.pipeline import make_pipe создаем объект который упрощает сборку за это отвечает параметр *steps*, который по кирпичику выстраивает последовательность наших действий  в нашем случае это процесс Маштабирование, Обучение и Предсказания:

pipe.fit(X_test)
pipe.predict(X_test)

Для создания сетки и нахождения в ней лучшего Accuracy по разным моделям, весам и количесва соседей используется GridSearchCV
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

С помощь параметров: *estimator, parm_grid, scoring, cv, n_jobs, refit* находим лучшие grid.best_score_ и grid.best_params_ 

Лучшие параметры: {'knn__metric': 'euclidean', 'knn__n_neighbors': 5, 'knn__weights': 'uniform', 'normalizer': 'passthrough'}
Лучшая CV accuracy: 0.9666666666666668
Test accuracy: 0.9666666666666667

### 1.6 Работа с Классификацией текстов с kNN
Так, работать с числами мы научились. Плохо, но научились, и впрос этого задания в том, а как работать когда чисел нет и есть только слова...

target в этой задаче — это **класс (тема) текста**, то есть _какой из newsgroups принадлежит документ_.
Вводные данные (для меня они были неочевидны):

target, наш Y
Это **метки**, они не требуют никаких преобразований.  
Они нужны, чтобы модель училась: “вот текст → вот тема”.

X переводи в Вектора (признаки X_vectorized)

Это **числовое представление текста**, которое создаётся векторизатором:
- CountVectorizer
	https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
- TfidfVectorizer  

Векторизация применяется к X (текстам)

И сейчас в силу вступает перевод слов в векторы чисел, как по мне очень странные векторы, бинарные:
- каждый документ вектор
- каждое слово признак
И для этого помогают другие метрики:
#### Жаккар
Смысл: “сколько общих слов относительно всех уникальных слов”  
Это почти прямо моделирует идею: _похожие тексты имеют много общих слов._
#### Косинус
Смысл: “насколько похожи направления векторов”  
Это хорошо работает, когда тексты разной длины: большой текст не должен автоматически быть “дальше”.

**Главная цель сравнения**: увидеть, что **одна и та же модель kNN** может резко менять качество в зависимости от метрики.

Меня обманули с примером. Думал, что 22 строка - это неверно определенный класс:
	![[Pasted image 20260203194831.png]]