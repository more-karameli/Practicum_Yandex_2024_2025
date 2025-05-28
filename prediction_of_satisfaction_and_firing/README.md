Проект был выполнен в ходе курса Data Science от Яндукс.Практикума.

Была предоставилена задача для поиска путей оптимизации управления персоналом для избегания оттока сотрудников для бизнеса. В рамках задачи было выделено два этапа: предсказание уровня удовлетворенности сотрудниками компанией, и второй этап - предсказание увольнения сотрудников и анализ факторов, которые на это влияют. 

1. Для решения первой задачи был создан пайплайн, включающий в себя предобработку данных и обучение нескольких моделей для задачи регрессии. Перебор моделей и гиперпараметров был выполнен с помощью случайного поиска с кросс-валидацией. Для обучения была создана отдельная метрика: симметричное среднее абсолютное процентное отклонение (SMAPE), который показывает, на сколько процентов отклоняются предсказанные значения относительно истинных в среднем, для лучшей модели этот показатель был 11.3% на тестовых данных. Лучшей моделью же оказался Градиентный Бустинг для регрессии.

2. Для предсказания оттока, а именно - уволится сотрудник или нет, были использован так же пайплайн, включающий в себя предобработку данных из предыдущей задачи (учитывая, что признаки входные те же, это стандартизирует подход) и перебор моделей классификации. Кроме того, для классификации был использован новый признак - предсказанные значения удовлетворенности работой. Перебор моделей и гиперпараметров был так же выполнен с использованием рандомизированного поиска с кросс-валидацией и учитыванием дисбаланса классов с помощью взвешивания классов внутри моделей. Лучшей моделью для данной задачи оказалось Дерево решений. 

3. Был выполнен дополнительно анализ влияния признаков на предсказание классов по SHAP. 

В работе использовались следующие библиотеки:

```python
# для работы с данными и визуализации
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import phik

# для создания пайплайна
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# для преобразования данных
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler 
from sklearn.impute import SimpleImputer

# для перебора моделей и параметров
from sklearn.model_selection import RandomizedSearchCV

# модели задачи 1
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

# метрики задачи 1
from sklearn.metrics import root_mean_squared_error, make_scorer

# для 2-ой задачи

# модели
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# метрики
from sklearn.metrics import f1_score, roc_auc_score

# для анализа признаков
import shap
```