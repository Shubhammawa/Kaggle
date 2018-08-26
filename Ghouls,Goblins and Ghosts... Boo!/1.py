import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

data_train = pd.read_csv("train.csv").values
data_test = pd.read_csv("test.csv").values

#print(data_train)
X = data_train[:,1:5]
Y = data_train[:,6]


#print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
#plt.hist(X_train[:,1])
#plt.show()

clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)

print(metrics.accuracy_score(Y_test,Y_pred))