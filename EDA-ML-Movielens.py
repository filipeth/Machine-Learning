import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics

plt.style.use('ggplot')
desired_width = 400
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

dat_movie = os.path.join('/Volumes', 'Filipe Theodoro', 'Artificial Inteligence', 'Projects', 'Projects for Submission'
                         , 'Project4_Movielens', 'movies.dat')
dat_users = os.path.join('/Volumes', 'Filipe Theodoro', 'Artificial Inteligence', 'Projects', 'Projects for Submission'
                         , 'Project4_Movielens', 'users.dat')
dat_rating = os.path.join('/Volumes', 'Filipe Theodoro', 'Artificial Inteligence', 'Projects', 'Projects for Submission'
                          , 'Project4_Movielens', 'ratings.dat')

df_movie = pd.read_csv(dat_movie, sep='::', engine='python', names=['MovieID', 'Title', 'Genres'], header=None)
df_user = pd.read_csv(dat_users, sep='::', engine='python', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
                      header=None)
df_rating = pd.read_csv(dat_rating, sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                        header=None)

df_movie['Overall rate'] = 0

plt.figure(figsize=(10,8))
df_user.Age.value_counts().plot(kind='barh')
plt.xlabel('Countsa')
plt.title('Age distribution')
plt.show()

df_rating.sort_values(by='MovieID', inplace=True)
df_movie['Overall rate'] = df_movie['MovieID'].map({num: (df_rating.loc[df_rating['MovieID'] == num]['Rating'].mean()) for num in df_movie.MovieID})

plt.figure(figsize=(10,8))
plt.scatter(df_movie.MovieID, df_movie['Overall rate'], color='b', edgecolors='k')
plt.xlabel('Movie Id')
plt.ylabel('Rating')
plt.title('Overall rating per movie')
plt.show()

df_toystory = pd.DataFrame(df_rating.loc[df_rating['MovieID'] == 1]['UserID'], columns=['UserID'])
df_toystory['Age'] = df_toystory['UserID'].map({num: (df_user.loc[df_user['UserID'] == num, 'Age'].item()) for num in df_toystory.UserID})

plt.figure(figsize=(8,8))
df_toystory.Age.value_counts().plot(kind='barh')
plt.title('Toystory age viewership')
plt.xlabel('Number count')
plt.show()

df_movie.sort_values(by='Overall rate', inplace=True, ascending=False)
top = df_movie[0:25]['MovieID'].values
teste = df_rating[df_rating['MovieID'].isin(top)].drop(['UserID', 'Timestamp'], axis=1)

plt.figure(figsize=(8,8))
teste.Rating.value_counts().plot(kind='barh')
plt.title('Top 25 movies by viewership rating')
plt.xlabel('Number count')
plt.show()

part_user = df_rating[df_rating['UserID'] == 2696].copy()
part_user['MovieID'] = part_user['MovieID'].map({num: (df_movie.loc[df_movie['MovieID'] == num, 'Title'].item()) for num in part_user.MovieID})

plt.figure(figsize=(8,8))
plt.scatter(part_user.Rating, part_user.MovieID)
plt.subplots_adjust(left=.46)
plt.xlabel('Rating')
plt.ylabel('Movie')
plt.title('Rating data by user of user id = 2696')
plt.show()

df_test = df_rating.drop(['Timestamp'], axis=1)
df_test = df_test.merge(df_user, on='UserID').drop(['Gender', 'Zip-code', 'UserID'], axis=1)
# print(df_test.head())
df_test = df_test.sample(frac=1)
x = df_test.drop('Rating', 1).values
y = df_test['Rating'].values
print(df_test.head())
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(x)
# scaled_df = pd.DataFrame(scaled_df, columns=df_test.columns)
# print(scaled_df[0:5])
# print(y[0:5])

x_train = scaled_df[0:500]
y_train = y[0:500]
x_test = scaled_df[-10]
print(x_test)
logReg = LogisticRegression()
logReg.fit(x_train, y_train)
y_pred = logReg.predict([x_test])
print('Predicted', y_pred)
print('Expected ', y[-10])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)
logReg = LogisticRegression()
logReg.fit(x_train, y_train)
y_pred = logReg.predict(x_test)
print('Test two', metrics.accuracy_score(y_test, y_pred))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print('Test three', metrics.accuracy_score(y_test, y_pred))

lineRg = LinearRegression()
lineRg.fit(x_train, y_train)
y_pred = lineRg.predict(x_test)
print('Test four', metrics.accuracy_score(y_test, y_pred.round()))

plt.figure(figsize=(8,8))
plt.subplots_adjust(hspace=.5)
plt.subplot(3, 1, 1)
plt.hist(df_test.MovieID, bins='auto')
plt.xlabel('MovieID')
plt.title('Movie Histogram')

plt.subplot(3,1,2)
plt.hist(df_test.Occupation, bins=20)
plt.xlabel('Occupation')
plt.title('Occupation Histogram')

plt.subplot(3,1,3)
plt.hist(df_test.Age, bins=7)
plt.xlabel('Age')
plt.title('Age Histogram')
plt.show()



