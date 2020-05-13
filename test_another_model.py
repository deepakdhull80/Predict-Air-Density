"""Decision tree Regressor

"""

import numpy as np
import pandas as pd
import seaborn as ss
import os
from sklearn import metrics
os.chdir("D:/python programs/machine learning/air quality index")


data=pd.read_csv("data/aqi-data.csv")
data.dropna()

X=data.iloc[:,:8]
Y=data['PM 2.5']
Y=np.expand_dims(Y,axis=1)

X=X.drop(columns=['SLP'])
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(X,Y,shuffle=True,test_size=0.2)



def norm(x):
    return ((x-np.mean(x))/np.std(x))


train_x=np.nan_to_num(train_x)
train_y=np.nan_to_num(train_y)

test_x=np.nan_to_num(test_x)
test_y=np.nan_to_num(test_y)

train_x=norm(train_x)
test_x=norm(test_x)



from sklearn.tree import DecisionTreeRegressor

model= DecisionTreeRegressor()

model.fit(train_x,train_y)


predict=model.predict(test_x)

import matplotlib.pyplot as plt


plt.plot(test_y[40:80])
plt.plot(predict[40:80])

print('MAE:', metrics.mean_absolute_error(test_y, predict))
print('MSE:', metrics.mean_squared_error(test_y, predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(test_y, predict)))








