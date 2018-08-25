from sklearn import svm
from sklearn import metrics
import pandas as pd
from sklearn.cross_validation import train_test_split

data_train=pd.read_csv("train.csv").as_matrix()
data_test=pd.read_csv("test.csv").as_matrix()

X = data_train[:,1:]
y = data_train[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = svm.SVC()
clf.fit(X_train,y_train)

#print(clf.predict(data_test))
y_pred = clf.predict(X_test)

print(metrics.accuracy_score(y_test,y_pred))