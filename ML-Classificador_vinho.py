import time
import random
from math import *
import operator
import pandas as pd
import numpy as np
import os.path


# import plotting libraries
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

import seaborn as sns
sns.set(style="white", color_codes=True)
sns.set(font_scale=1.5)

# import the ML algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from statsmodels.tools.eval_measures import rmse

# pre-processing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer
from sklearn.linear_model import LogisticRegression

# import libraries for model validation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# import libraries for metrics and reporting
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics

desired_width=400
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

plt.style.use('ggplot')
csv_file = os.path.join('/Users', 'filipetheodoro', 'Downloads',
                        "Curso", 'datasets', 'winequality-data.csv')
df_wine = pd.read_csv(csv_file)
print(df_wine.head())
# print(df_wine['quality'].value_counts())
x = df_wine.drop('quality', 1).values
y1 = df_wine['quality'].values
y = y1 <= 5
teste = df_wine.sort_values(by='quality', ascending=False)
# print(teste.head())

# pd.DataFrame.hist(df_wine, figsize=[10, 10])
# plt.subplots_adjust(hspace=0.75)
# plt.show()

# plt.figure(figsize=(7, 7))
# plt.hist(y1, bins=7)
# plt.xlabel('Quality')
# plt.ylabel('number of wines')
# plt.show()

# plt.figure(figsize=(20,5))
# plt.subplot(1, 2, 1)
# plt.hist(y1, bins=7)
# plt.xlabel('original target value')
# plt.ylabel('count')
# plt.show()
# plt.subplot(1, 2, 2)
# plt.hist(y)
# plt.xlabel('aggregated target value')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
knn_model_1 = knn.fit(x_train, y_train)

linreg1 = LogisticRegression()
linreg1.fit(x_train, y_train)
y_pred1 = linreg1.predict(x_test)

print('k-NN accuracy for test 1 set: %.2f' % knn_model_1.score(x_test, y_test))
print('Lin Reg accuracy for test 1 set: %.2f' % metrics.accuracy_score(y_test, y_pred1))
# print('knn ', metrics.accuracy_score(y_test, x_pred))
# y_true, y_pred1 = y_test, knn_model_1.predict(x_test)
# print(classification_report(y_true, y_pred1))

xs = scale(x)

xs_train, xs_test, ys_train, ys_test = train_test_split(xs, y, test_size=0.1, random_state=42)
knn_model_2 = knn.fit(xs_train, ys_train)

linreg2 = LogisticRegression()
linreg2.fit(xs_train, ys_train)
y_pred2 = linreg1.predict(x_test)

print('k-NN accuracy for test 2 set: %.2f' % knn_model_2.score(xs_test, ys_test))
print('Lin Reg accuracy for test 2 set: %.2f' % metrics.accuracy_score(y_test, y_pred2))

# ys_true, ys_pred = ys_test, knn_model_2.predict(xs_test)
# print(classification_report(ys_true, ys_pred))






