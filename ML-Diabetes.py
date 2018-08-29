import pandas as pd
import numpy as np
import os.path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cluster import KMeans
csvfile = os.path.join('/Volumes', 'Filipe Theodoro', 'Artificial Inteligence', 'Projects', 'assigment 6', 'pima-indians-diabetes.data')
df = pd.read_csv(csvfile, header=None)
print(df.shape)
df.columns = ['Times pregnant', 'Glucose concentration', 'Blood pressure', 'Skin fold thickness',
                                  'Serum insulin', 'Body mass index', 'Diabetes pedigree function', 'Age', 'Class']
print(df.head())
x_features = df[['Times pregnant', 'Glucose concentration', 'Blood pressure', 'Skin fold thickness',
                                  'Serum insulin', 'Body mass index', 'Diabetes pedigree function', 'Age']]
y_target = df['Class']
x_train, x_test, y_train, y_test = train_test_split(x_features, y_target, random_state=1)
linreg = LogisticRegression()
linreg.fit(x_train, y_train)
print(linreg.intercept_, linreg.coef_)
y_pred = linreg.predict(x_test)
print(y_pred)
print(metrics.accuracy_score(y_test, y_pred))
print('actual:     ', y_test.values[0:30])
print('calculated: ', y_pred[0:30])


# print('Linear Regression error',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#
# predict_y = KMeans(n_clusters=2, random_state=0).fit_predict(x_train)
# print(metrics.accuracy_score(y_train, predict_y))
# print(predict_y.shape, y_train.shape)
# print('KNN error', np.sqrt(metrics.mean_squared_error(predict_y, y_train)))


