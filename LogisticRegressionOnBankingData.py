import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
df=pd.read_csv("C:/Users/Yogesh Kumar Ahuja/Desktop/data2.csv")
# print(df['education'].unique())
df['education']=np.where(df['education']=="basic.9y","Basic",df['education'])
df['education']=np.where(df['education']=="basic.6y","Basic",df['education'])
df['education']=np.where(df['education']=="basic.4y","Basic",df['education'])
# print(df['education'].unique())
# print(pd.get_dummies(df['education'],prefix='yogesh'))
print(df['y'].value_counts())
sns.countplot(x='y',data=df,palette=None)
# plt.show()
# plt.savefig("countplo")
count_no=len(df[df['y']==0])
count_yes=len(df[df['y']==1])

perno=(count_no/(count_no+count_yes))*100
peryes=(count_yes/(count_no+count_yes))*100
print(perno)
print(peryes)
# print(df.groupby('marital').mean())
# pd.crosstab(df.job,df.y).plot(kind='bar')
# pd.crosstab(df.marital,df.y).plot(kind='bar')
# pd.crosstab(df.poutcome,df.y).plot(kind='bar')
# plt.show()
cat_var=['job','marital','education','contact','month','day_of_week','poutcome']
for i in cat_var:
    # cat_list='var'+'_'+ i
    cat_list=pd.get_dummies(df[i],prefix=i)
    # print(cat_list)
    df1=df.join(cat_list)
    print(df1)
#
# dict={1:"rahul",2:"sanjay",3:"ajay"}
# print(dict.get(1))
# for i in dict:
#     print(dict.get(i))