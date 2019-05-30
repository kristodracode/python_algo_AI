import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

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
    df=df1
    # print(df.columns.values)
    # print(df)

# print(df)
# print(df.columns.values)
cat_vars=['job','marital','education','contact','month','day_of_week','poutcome','default','housing','loan']
data_vars=df.columns.values.tolist()
w=[i for i in data_vars if i not in cat_vars]
data_final=df[w]
# print(data_final.columns.values)
# pd.crosstab(data_final.age,data_final.marital_single).plot(kind='line')
# plt.show()
#
Y=['y']
data_varsy=data_final.columns.values.tolist()
x=[i for i in data_varsy if i not in Y]
data_finalx=df[x]
print(data_finalx.columns.values)


X=data_final.loc[:, data_final.columns!='y']
Y=data_final.loc[:, data_final.columns=='y']
print(Y)
print("This is x",X)
os=SMOTE(random_state=0)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
col=X_train.columns
print("this is  xtrain",X_train)
print("this is  xtest",X_test)
print("this is  ytrain",Y_train)
print("this is  ytest",Y_test)
os_data_X,os_data_Y=os.fit_sample(X_train,Y_train.values.ravel())
# =os.fit_sample(X_train,Y_train)
print("This is os vala datax",os_data_X)
print("Line ends here")
print("This is os vala datay",os_data_Y)
print("Line ends here")
# os_data_X=pd.DataFrame(data=os_data_X,columns=col)
# os_data_Y=pd.DataFrame(data=os_data_Y,columns='y')
# print(len(os_data_X))


lr=LogisticRegression()
qu=lr.fit(X_train,Y_train.values.ravel())
rfe=RFE(lr,500)
print(rfe)
print(qu)

Y_predict=lr.predict(X_test)
print(Y_predict)
print('{:.2f}'.format(lr.score(X_test,Y_test)))

