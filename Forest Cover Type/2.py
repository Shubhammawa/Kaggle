import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import time

data_train = pd.read_csv("train.csv").values
data_test = pd.read_csv("test.csv").values

###-----------------Modelling phase-------------------------###

X = data_train[:,1:55].astype(float)
Y = data_train[:,55].astype(float)
#print(X.shape)
# for i in range(0,55):
# 	print(np.mean(X[:,i]))
# 	sns.distplot(X[:,i])
# 	plt.show()
# 	plt.close()

# Features 1-10 are make a dense matrix, are not centred at zero
# Other features make a sparse matrix - binary values {0,1}

X_scaled = preprocessing.scale(X[:,0:10])

# for i in range(0,10):
# 	print(np.mean(X_scaled[:,i]))
# 	sns.distplot(X_scaled[:,i])
# 	plt.show()
# 	plt.close()

# print(X_scaled[0:10,9])
# print(np.mean(X_scaled,axis=0))
# print(np.std(X_scaled,axis=0))

X[:,0:10] = X_scaled[:,0:10]
#print(X[0:15,:])

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

### Model 1-------------------------------------------
clf_1 = RandomForestClassifier(n_estimators=100)
clf_1.fit(X_train,Y_train)
Y_pred = clf_1.predict(X_test)
X_pred = clf_1.predict(X_train)
print("Training accuracy: ",metrics.accuracy_score(Y_train,X_pred))
print("Testing accuracy: ",metrics.accuracy_score(Y_test,Y_pred))
# 81.79
###----------------------------------------------------

### Model 2--------------------------------------------
# clf_2 = LogisticRegression()
# clf_2.fit(X_train,Y_train)
# Y_pred = clf_2.predict(X_test)

# print(metrics.accuracy_score(Y_test,Y_pred))
# 68.59
###----------------------------------------------------

### Model 3--------------------------------------------
# clf_3 = LinearSVC(max_iter=100)
# clf_3.fit(X_train,Y_train)
# Y_pred = clf_3.predict(X_test)

# print(metrics.accuracy_score(Y_test,Y_pred))
# 66.37
###----------------------------------------------------

### Model 4--------------------------------------------

# clf_4 = SGDClassifier(max_iter=1000,learning_rate='invscaling',eta0=0.1)
# clf_4.fit(X_train,Y_train)
# Y_pred = clf_4.predict(X_test)
# X_pred = clf_4.predict(X_train)
# print(metrics.accuracy_score(Y_train,X_pred))
# print(metrics.accuracy_score(Y_test,Y_pred))

### Model 4--------------------------------------------
# clf_5 = MLPClassifier(batch_size=32,learning_rate_init=0.01,max_iter=200,tol=0.00000000001,verbose=True)
# clf_5.fit(X_train,Y_train)
# Y_pred = clf_5.predict(X_test)
# X_pred = clf_5.predict(X_train)
# print("Training accuracy: ",metrics.accuracy_score(Y_train,X_pred))
# print("Testing accuracy: ",metrics.accuracy_score(Y_test,Y_pred))
###-----------------------------------------------------

