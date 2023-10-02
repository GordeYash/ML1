import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

df=pd.DataFrame(pd.read_csv('FDS Practice/Employee.csv'))
print(df.head(5))

#Data Cleaning and information gathring

print(df.isnull().sum())
print(df.dtypes)



#Cheking catogorical data
print(df['PaymentTier'].value_counts())
df.drop(['City','Gender'],axis=1,inplace=True)


#Spliting Data into dependent and independent variables
x=df.iloc[:,0:6].values
y=df.iloc[:,-1].values
print(x[100])
print(y[100])

#Transforming Catogorical data to numerical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lb=LabelEncoder()
x[:,0]=lb.fit_transform(x[:,0])

lb1=LabelEncoder()
x[:,-2]=lb1.fit_transform(x[:,-2])

print(x.shape)
print(y.shape)

#spliting data into training and testing
from sklearn.model_selection import train_test_split 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

print(x_train[0,:])

#Applying random forest algoritham

#69 % acuuracy
#from sklearn.svm import SVC
#cif=SVC(kernel='linear')
#cif.fit(x_train,y_train)
#y_pred=cif.predict(x_test)

#80 % accuracy
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='gini',random_state=0)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

#70 % accuracy
#from sklearn.neighbors import KNeighborsClassifier
#classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
#classifier.fit(x_train,y_train)
#y_pred=classifier.predict(x_test)

print(x_test[100])

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_pred,y_test)

acc=accuracy_score(y_pred=y_pred,y_true=y_test)
print(acc*100)


feture=[[1,2018,0,19,1,7]]
print(np.shape(feture))
predition=classifier.predict(feture)
#print(predition)

pickle.dump(classifier,open("model.pkl","wb"))
pickle.dump(lb,open('lb.pkl','wb'))
pickle.dump(lb1,open('lb1.pkl','wb'))

