from sklearn.datasets import load_iris
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris=pd.read_csv("C:/Users/Yogesh Kumar Ahuja/Desktop/Iris.csv")
# print(type(iris))
# print(iris.data)
# print(iris.feature_names)
# print(iris.target)
# print(iris.target_names)
# print(type(iris.data))
# print(type(iris.target))
# print(iris.data.shape)
x=iris.iloc[:,:4].values
# print(x)
y=iris.iloc[:,4].values
# print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(y_train)
print(x_test)

print(y_test)
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y_train)
y_predict=classifier.predict(x_test)

print("yhan tak sahi hua")
print(y_predict)
print("ye y test hai")
print(y_test)