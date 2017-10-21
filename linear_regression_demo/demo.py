import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data
dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()




## simple n network on sigmoid
## activation function sigmoid(x)=1/(1+e(−x))​
## output of network (2 parameters and bias, like linear regression):
## y=f(h)=sigmoid(∑ w*x + b)
## 
import numpy as np

def sigmoid(x):
    # TODO: Implement sigmoid function
    return 1 / ( 1 + np.exp(-x))

inputs = np.array([0.7, -0.3])
weights = np.array([0.1, 0.8])
bias = -0.1

# TODO: Calculate the output
output = sigmoid(np.dot(weights, inputs) + bias)

print('Output:')
print(output)
