# WineReview

___
### Выполнили:
## Мингачев Рустам, Гатин Рамиль 11-003
___

### Streamlit

<https://ramilgatin-winereview-main-kj1k3w.streamlit.app/>

___
### Папка GoogleDrive (датасет)
https://drive.google.com/drive/folders/1Hxe6E9TuzWrpCvI3GJyKSXdtCf-ccqKd

https://www.kaggle.com/datasets/zynicide/wine-reviews

___
### Видео с защитой проекта

https://www.youtube.com/watch?v=SLnsfTCnY58&ab_channel=PioneeR

### Описание датасета

Этот набор данных содержит три файла:

winemag-data-130k-v2.csv содержит 10 столбцов и 130 тысяч строк обзоров вин.
___
### Задача

Создать модель, которая может идентифицировать сорт, винодельню и местоположение вина на основе описания.
___
### Реализованные методы

*Отбор признаков для уменьшения размерности ДатаСета, используя CatBoostRegressor

Отбор признаков (feature selection) – это оценка важности того или иного признака с помощью алгоритмов машинного обучения и отсечение ненужных.

*Прогнозирование баллов на основе описания (NLP)

NLP — одно из направлений искуственного интеллекта, которое работает с анализом, пониманем и генерацией живых языков, для того, чтобы взаимодействовать с компьютерами и устно, и письменно, используя естественные языки вместо компьютерных.

___
### Базовые понятия

Стеммизация — процесс приведения слова к его корню/основе.

Стоп-слова — это часто используемые слова, которые не вносят никакой дополнительной информации в текст. Слова типа "the", "is", "a" не несут никакой ценности и только добавляют шум в данные.
___
### Результаты

Feature selection - используя данный метод, мы практически не изменили точность работы, но экономим некоторое вычислительное время и оптимально используем оперативную память.

NPL - Оценки ценности с помощью Bag of Words Counts и TF-IDF technique практически не различаются на основе данных по признаку description

