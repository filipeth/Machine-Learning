import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import pickle as pkl
from sklearn import metrics

plt.style.use('ggplot')
desired_width = 400
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

csv_test = os.path.join('/Volumes', 'Filipe Theodoro', 'Artificial Inteligence', 'Projects', 'Projects with Solution',
                        'Project 2', 'Datasets', 'test.csv')
csv_train = os.path.join('/Volumes', 'Filipe Theodoro', 'Artificial Inteligence', 'Projects', 'Projects with Solution',
                         'Project 2', 'Datasets', 'train.csv')

df_train = pd.read_csv(csv_train)
df_test = pd.read_csv(csv_test)
print(df_test.PassengerId.max())

# print(df_train.head())
# print(df_train.Sex.value_counts())
# print(df_train.Age.describe())

# plt.figure(figsize=(10,8))
# df_train.Age.hist(bins=70, rwidth=0.8)
# plt.xlabel('Age')
# plt.ylabel('Number of occurrences')
# plt.title('Age distribution')
# plt.show()
#
# plt.figure(figsize=(10,8))
# plt.scatter(df_train.Survived, df_train.Age)
# plt.xlabel('Survival (Died = 0)')
# plt.ylabel('Age')
# plt.show()

# plt.figure(figsize=(8,8))
# df_train.Survived.value_counts().plot(kind='barh')
# plt.title('Survival Breakdown (Died = 0, Survived = 1)')
# plt.xlabel('Number count')
# plt.show()

# print(df_train.Pclass.value_counts())
# plt.figure(figsize=(8,8))
# df_train.Pclass.hist(bins='auto')
# plt.xlabel('Class')
# plt.ylabel('Value count')
# plt.title('Class distribution')
# plt.show()

# print(df_train.Embarked.value_counts())
# plt.figure(figsize=(10,8))
# df_train.Embarked.value_counts().plot(kind='barh')
# plt.title('Embarkation location')
# plt.xlabel('Number of occurrences')
# plt.show()


def cat_to_num(df):
    # df_train.Sex = np.where(df_train.Sex == 'male', 0, 1) # MALE=0 FEMALE=1
    df.replace({'Embarked': {'S': 0, 'C': 1, 'Q': 2}, 'Sex': {'male': 0, 'female': 1}}, inplace=True)
    df.Age = df.Age.fillna(df.Age.mean())
    df.Fare = df.Fare.fillna(df.Fare.mean())
    return df


df_train = df_train.dropna(axis=0, how='any')
x_train = cat_to_num(df_train).drop(['Survived', 'Ticket', 'Name', 'Cabin'], 1).values
x_test = cat_to_num(df_test).drop(['Ticket', 'Name', 'Cabin'], 1).values
model = LogisticRegression()
model.fit(x_train, df_train.Survived)
predicted = model.predict(x_test)
# print(predicted)

persist_model = pkl.dumps(model)

# Save model as a file
joblib.dump(model, 'LogRegModel.pkl')

# Load model from file
new_model = joblib.load('LogRegModel.pkl')

