import numpy as np
import pandas as pd
# from sklearn.datasets import load_boston
#
# boston_data = load_boston()
# print(boston_data['DESCR'])
# print(boston_data['feature_names'])
#
# df_boston = pd.DataFrame(boston_data.data)
# df_boston.columns = boston_data.feature_names
# print(df_boston.head())
# print(df_boston.shape)
# print(boston_data.target.shape)
# print(boston_data['target'])
# df_boston['Price'] = boston_data.target
# print(df_boston.head())
# x_features = boston_data.data
# y_target = boston_data.target
# from sklearn.linear_model import LinearRegression
#
# lineRg = LinearRegression()
# lineRg.fit(x_features, y_target)
# print('the estimated intercept', lineRg.intercept_)
# print('the coefficient is', len(lineRg.coef_))
# from sklearn import model_selection
# x_train, x_test, y_train, y_test = model_selection.train_test_split(x_features, y_target)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# lineRg.fit(x_train, y_train)
# print('mse value is', np.mean((lineRg.predict(x_test) - y_test) ** 2))
# print('variance is', lineRg.score(x_test,y_test))
# from sklearn.datasets import load_iris
# from sklearn.neighbors import KNeighborsClassifier
# iris_data = load_iris()
# print(type(iris_data))
# print(iris_data.DESCR)
# print(iris_data.feature_names)
# print(iris_data.target)
# print(iris_data.data.shape)
# x_feature = iris_data.data
# y_target = iris_data.target
# print(x_feature.shape)
# print(y_target.shape)
# df = pd.DataFrame(x_feature, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
# df['Type'] = y_target
# print(df)
# knn = KNeighborsClassifier(n_neighbors=1)
# print(knn)
# knn.fit(x_feature, y_target)
# x_new = [[3,5,4,1], [5,3,4,2]]
# print(knn.predict(x_new))
# from sklearn.linear_model import LogisticRegression
# logReg = LogisticRegression()
# logReg.fit(x_feature, y_target)
# print(logReg.predict(x_new))


# from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
#
# n_samples = 300
# random_state = 20
# x, y = make_blobs(n_samples=n_samples, n_features=5, random_state=None)
# predict_y = KMeans(n_clusters=3, random_state=random_state).fit_predict(x)
# print(predict_y)

# from sklearn.decomposition import PCA
# n_samples = 20
# random_state = 20
# x,y = make_blobs(n_samples=n_samples, n_features=10, random_state=None)
# print(x.shape)
# pca = PCA(n_components=3)
# pca.fit(x)
# print(pca.explained_variance_ratio_)
# first_pca = pca.components_[0]
# print(first_pca)
# pca_reduced = pca.transform(x)
# print(pca_reduced.shape)

# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
# from sklearn.decomposition import PCA
#
# estimator = [('dim_reduction', PCA()), ('logres_model', LogisticRegression())]
# # print(estimator)
# pipeline_estimator = Pipeline(estimator)
# # print(pipeline_estimator.steps[0])
# print(pipeline_estimator.steps)

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle as pkl
from sklearn.externals import joblib
iris_datset = load_iris()

print(iris_datset.feature_names)
print(iris_datset.target)
x_feature = iris_datset.data
y_target = iris_datset.target
x_new = [[3,5,4,1], [5,3,4,2]]
logreg = LogisticRegression()
logreg.fit(x_feature, y_target)
print(logreg.predict(x_new))
persist_model = pkl.dumps(logreg)
joblib.dump(logreg, 'regresfilename.pkl')
new_logreg = joblib.load('regresfilename.pkl')
print(new_logreg.predict(x_new))

