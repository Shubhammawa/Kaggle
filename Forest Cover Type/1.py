import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data_train = pd.read_csv("train.csv").values
data_test = pd.read_csv("test.csv").values

#print(data_train.shape)
X_train = data_train[:,1:55]
#print(data_train[0:5,55])
#print(X[242,53])	
Y_train = data_train[:,55]
X_test = data_test[:,1:55]
#print(X_test.shape)
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

# clf = RandomForestClassifier()
# clf.fit(X_train,Y_train)
# Y_pred = clf.predict(X_test)

# print(metrics.accuracy_score(Y_test,Y_pred))

clf = RandomForestClassifier()
clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)
#print(Y_pred.shape)
id = np.arange(15121,581013)
#print(id.shape)
#Y_pred = np.append(id,Y_pred,axis=1).reshape((565892,2))
#print(Y_pred)
Y_pred_final = np.zeros(shape=(565892,2))
Y_pred_final[:,0] = id.astype(int)
Y_pred_final[:,1] = Y_pred.astype(int)
#print(Y_pred_final)
pred = pd.DataFrame(Y_pred_final.astype(int))
pred.to_csv('submission.csv',header=['Id','Cover_Type'], index=False)