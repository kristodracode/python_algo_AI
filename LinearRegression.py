import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
df_train=pd.read_csv("C:/Users/Yogesh Kumar Ahuja/Desktop/ppts/random-linear-regression/train.csv")
df_test=pd.read_csv("C:/Users/Yogesh Kumar Ahuja/Desktop/ppts/random-linear-regression/test.csv")
x_train=df_train['x']
y_train=df_train['y']
x_test=df_test['x']
y_test=df_test['y']
x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)
x_train=x_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)
# print(x_test)
up=LinearRegression(normalize=True)
up.fit(x_train,y_train)
y_predict=up.predict(x_test)
print(y_predict)
# print(y_test)
print(r2_score(y_test,y_predict))

# plt.plot(x_test,y_predict)
# plt.scatter(x_test,y_test)
# plt.show()


#print(up)