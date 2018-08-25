import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib

data_train=pd.read_csv("train.csv").as_matrix()
#print(data)
data_test=pd.read_csv("test.csv").as_matrix()

#clf = DecisionTreeClassifier()

#X_train=data_train[0:30000,1:]
#y = data_train[0:30000,0]

X = data_train[:,1:]
y = data_train[:,0]

#clf.fit(X_train,y)
#clf.fit(X,y)

#joblib.dump(clf, 'Digit_weights1.pkl')
#X_test = data_train[30000:,1:]
#y_test = data_train[30000:,0]

#d = X_test[8]
#d.shape=(28,28)
#plt.imshow(255-d,cmap='gray')
#plt.show()

#y_pred = clf.predict(X_test)

#print(metrics.accuracy_score(y_test,y_pred))
#print(cross_val_score(clf, X, y, cv=10, scoring='accuracy', n_jobs=-1))

# Model trained and weights stored

# Loading weights to make predictions

clf = joblib.load('Digit_weights1.pkl')

X_test = data_test


#print(clf.predict(X_test))
pred = pd.DataFrame(clf.predict(X_test))
pred.index += 1
#pred.to_csv('submission.csv',header=['Label'],index_label='ImageId')

#d = X_test[0]
#d.shape=(28,28)
#plt.imshow(255-d,cmap='gray')
#plt.show()