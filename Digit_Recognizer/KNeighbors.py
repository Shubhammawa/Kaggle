import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split

data_train=pd.read_csv("train.csv").as_matrix()
data_test=pd.read_csv("test.csv").as_matrix()

X = data_train[:,1:]
y = data_train[:,0]

#X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = KNeighborsClassifier(n_neighbors=1, leaf_size=500, n_jobs=3)
#clf.fit(X_train,y_train)

#y_pred = clf.predict(X_test)

#print(metrics.accuracy_score(y_test,y_pred))

clf.fit(X,y)
#joblib.dump(clf, 'Random_forest.pkl')

#clf = joblib.load('Digit_weights1.pkl')

X_test = data_test

y_pred = pd.DataFrame(clf.predict(X_test))
y_pred.index += 1

y_pred.to_csv('submission7.csv',header=['Label'],index_label='ImageId')
