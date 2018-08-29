import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.datasets import load_boston
import pandas as pd
import seaborn as sns

# random = np.random.rand(10)
# print(random)
# web1 = [123, 645, 950, 1290, 1630, 1450, 1034, 1295, 465, 205, 80]
# web2 =[95,680,889,1145,1670,1323,1119,1265,510,310,110]
# web3 = [105,630,700,1006,1520,1124,1239,1380,580,610,230]
# time = list(range(7,18))
# print(time)


# style.use('ggplot')
# plt.plot(time, web1, 'g', label='monday', linewidth=2)
# v
# plt.plot(time, web3, 'b', label='wednesday', linewidth=2)
# plt.xlabel('Hrs')
# plt.axis([6.5,17.5,50,2000])
# plt.ylabel('Numbers of users')
# plt.title('Web traffic')
# plt.legend()
# plt.show()

# LINECHART
# plt.figure(figsize=(8,4))
# plt.subplots_adjust(hspace=1)
# plt.subplot(2,1,1)
# plt.title('Monday')
# plt.plot(time, web1, 'g', label='monday', linewidth=2, linestyle='--')
# plt.subplot(2,1,2)
# plt.title('Tuesday')
# plt.plot(time, web2, 'r', label='tuesday', linewidth=2, linestyle='--')
# plt.show()

# data = load_boston()
# x_axis = data.data
# y_axis = data.target


# HISTOGRAMS
# style.use('ggplot')
# plt.figure(figsize=(7,7))
# plt.hist(y_axis, bins=50)
# plt.xlabel('price in 1000s USD')
# plt.ylabel('number of houses')
# plt.show()


# SCATTERPLOT
# style.use('ggplot')
# plt.figure(figsize=(7,7))
# plt.scatter(x_axis[:,5], y_axis, color='b', edgecolors='k')
# plt.xlabel('price in 1000s USD')
# plt.ylabel('number of houses')
# plt.show()


# HEATMAPS
# flight_data = sns.load_dataset('flights')
# print(flight_data.head())
# flight_data = flight_data.pivot('month', 'year', 'passengers')
# print(flight_data)
# sns.heatmap(flight_data)
# plt.show()

# PIRCHARTS
# job_data = ['40', '20', '17', '8', '5', '10']
# labels = 'It', 'Finance', 'Marketing', 'Admin', 'Hr', 'Operations'
# explode = (0.05, 0, 0, 0, 0, 0)
# plt.pie(job_data, labels=labels, explode=explode)
# plt.show()

# ERRORBAR



