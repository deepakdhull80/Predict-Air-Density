"""
MAE: 44.364817800723955
MSE: 3688.7845628428327
RMSE: 60.7353650095464

"""




import keras
import numpy as np
import pandas as pd
import seaborn as ss
import os
from sklearn import metrics
os.chdir("D:/python programs/machine learning/air quality index")

data=pd.read_csv("data/aqi-data.csv")
data.dropna()

ss.pairplot(data)

data.describe()

row,col=data.shape
X=data.iloc[:,:8]
Y=data['PM 2.5']
Y=np.expand_dims(Y,axis=1)

X=X.drop(columns=['SLP'])
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(X,Y,shuffle=True,test_size=0.2)

corr=data.corr()
ss.heatmap(corr)



from sklearn.linear_model import LinearRegression

model=LinearRegression(fit_intercept=True,normalize=True)

def norm(x):
    return ((x-np.mean(x))/np.std(x))


train_x=np.nan_to_num(train_x)
train_y=np.nan_to_num(train_y)

test_x=np.nan_to_num(test_x)
test_y=np.nan_to_num(test_y)

train_x=norm(train_x)
test_x=norm(test_x)


model=model.fit(train_x,train_y)



predict_y=model.predict(test_x)

def loss(x,y):
    m=x.shape[0]
    return (1/m)*np.sum((x-y)**2)

def acc(x,y):
    m=x.shape[0]
    return (1/m)*np.sum(x==y)


print(loss(predict_y,test_y))


import matplotlib.pyplot as plt




plt.plot(test_y[40:80])
plt.plot(predict_y[40:80])

def go(x,y):
    
    print(model.predict([x]))
    
    print(y)

test_no=50
go(test_x[test_no,:],test_y[test_no])


#check

from sklearn import metrics


print('MAE:', metrics.mean_absolute_error(test_y, predict_y))
print('MSE:', metrics.mean_squared_error(test_y, predict_y))
print('RMSE:', np.sqrt(metrics.mean_squared_error(test_y, predict_y)))





