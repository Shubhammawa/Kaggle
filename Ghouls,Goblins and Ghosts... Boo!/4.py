import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import LabelEncoder

data_train = pd.read_csv("train.csv").values
data_test = pd.read_csv("test.csv").values

X = data_train[:,1:6]
Y = data_train[:,6]

##### Encoding color column #####
le1 = LabelEncoder()
le1.fit(X[:,4])
color = le1.transform(X[:,4])
color = color/5

##### Replacing color column in X with encodings ####
X[:,4] = color

##### Encoding Labels #####
le2 = LabelEncoder()
le2.fit(Y)
Y_enc = le2.transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y_enc)

print(X_train[0:5,:])
print(X_test[0:5,:])
print(Y_train[0:5])
print(Y_test[0:5])

#X_train = X
#Y_train = Y_enc

#X_test = data_test[:,1:6]
#Y_test = data_test[]

clf = tree.DecisionTreeClassifier()
clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)

clf2 = GradientBoostingClassifier(learning_rate=0.1,n_estimators=200,max_depth=3)
clf2.fit(X_train,Y_train)
Y_pred2 = clf2.predict(X_test)

clf3 = LogisticRegressionCV(multi_class='multinomial',max_iter=500,Cs=30)
clf3.fit(X_train,Y_train)
Y_pred3 = clf3.predict(X_test)

clf4 = SGDClassifier(alpha=0.02)
clf4.fit(X_train,Y_train)
Y_pred4 = clf4.predict(X_test)

clf5 = GaussianProcessClassifier()
clf5.fit(X_train,Y_train)
Y_pred5 = clf5.predict(X_test)

print(metrics.accuracy_score(Y_test,Y_pred2))

# clf5.fit(X,Y)
# Y_pred = clf5.predict()