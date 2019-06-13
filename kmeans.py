import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from numpy import matrix
df=pd.read_csv("C:/Users/Yogesh Kumar Ahuja/Desktop/data3.csv")
#m=[[71.24,28],[52.53,25],[64.54,27],[55.69,22],[54.58, 25]]
f1=df['Distance_Feature'].values
f2=df['Speeding_Feature'].values
from sklearn.svm import SVM


X=np.matrix(zip(f1,f2))
print(X)
km=KMeans(n_clusters=2).fit(X)
print(km)
# plt.scatter(df['Distance_Feature'],df['Speeding_Feature'])
# plt.show()
# print(km)