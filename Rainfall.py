import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv("austin_final.csv")
data

X = data.drop(['PrecipitationSumInches'], axis=1)

# the output or the label.
Y = data['PrecipitationSumInches']
# reshaping it into a 2-D vector
Y = Y.values.reshape(-1, 1)

# consider a random day in the dataset
# we shall plot a graph and observe this day
day_index = 798
days = [i for i in range(Y.size)]

clf = LinearRegression()
# train the classifier with our input data
clf.fit(X, Y)

inp = np.array([[74], [60], [45], [67], [49], [43], [33], [45],
				[57], [29.68], [10], [7], [2], [0], [20], [4], [31]])
inp = inp.reshape(1, -1)

print('The precipitation in inches for the input is:', clf.predict(inp))

print("the precipitation trend graph: ")
plt.scatter(days, Y, color='g')
plt.scatter(days[day_index], Y[day_index], color='r')
plt.title("Precipitation level")
plt.xlabel("Days")
plt.ylabel("Precipitation in inches")

plt.show()
x_vis = X.filter(['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent',
				'SeaLevelPressureAvgInches', 'VisibilityAvgMiles',
				'WindAvgMPH'], axis=1)

# plot a graph with a few features (x values)
print("Precipitation vs selected attributes graph: ")

for i in range(x_vis.columns.size):
	plt.subplot(3, 2, i + 1)
	plt.scatter(days, x_vis[x_vis.columns.values[i][:100]],
				color='b')

	plt.scatter(days[day_index],
				x_vis[x_vis.columns.values[i]][day_index],
				color='r')

	plt.title(x_vis.columns.values[i])
plt.show()

