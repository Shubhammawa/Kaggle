import numpy as np 
from scipy.optimize import minimize
import csv

def sigmoid(x):
        sigmoid = 1/(1+np.exp(-x))
        return sigmoid
#n = float(input('Enter a number'))
#print(sigmoid(n))	
def prediction(h):										
# Predicts the final outcome
        predict = []
        for i in h:
                if(i>=0.5):
                        predict.append(1)
                else:
                        predict.append(0)
        return predict
def hypothesis(x,w = np.array([1,1,1,1,1,1,1])):
# Computes the hypothesis function	        
        x = np.transpose(x)
        h = []
        f = np.multiply(x,w)
        for i in f:
                h.append(sigmoid(f))
        h_w = [h,w]
        return h_w
def cost(y,h):
# Computes the cost of the current hypothesis function
        cost = 0
        for i in range(0,len(y)):
                cost+= -y[i]*np.log(h[i]) - (1-y[i])*np.log(1-h[i])
        j = cost/len(y)
        return j

def train(j,w):
# Finds the optimal weights to minimize the cost function
# w - weights
        solution = minimize(j,w)	
        w = solution.x
        return w
def import_data():
        ldata = []
        with open('train.csv') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    ldata.append(row)
        data = np.array(ldata)
        return data                     # data[0] = labels           
                                               # data[i] = record of ith passenger
                                               # data[1:,i] = array of ith feature (containing feature - value for all passengers), ex : data[1:,0] = array of passenger_id
#print(import_data()[0])
#['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare' 'Cabin' 'Embarked']
def data_preprocessing(data):
        y = data[1:,1]                  # labels

# Filling of missing values

# data missing for age, embarked
# For age : replacing missing values with mean age 
        sum = 0
        for i in range(1,len(data)):
                if(data[i,5] != ''):
                        sum = sum + float(data[i,5])        
        mean_age = sum/(len(data)-1)

        for i in range(1,len(data)):
                if(data[i,5] == ''):
                        data[i,5] = mean_age
# Mapping ['male', 'female'] to [0,1]
        for i in range(1,len(data)):
                #print(i)
                if(data[i,4] == 'male'):
                        data[i,4] = 0
                elif(data[i,4] == 'female'):
                        data[i][4] = 1

# Mapping all string inputs to appropriate data types
        for i in range(1,len(data)):
                data[i,2] = int(data[i,2])
                data[i,5] = float(data[i,5])
                data[i,6] = int(data[i,6])
                data[i,7] = int(data[i,7])
                data[i,9] = float(data[i,9])
                if(data[i,11] == 'C'):
                        data[i,11] = 0
                elif(data[i,11] == 'Q'):
                        data[i,11] = 1
                elif(data[i,11] == 'S'):
                        data[i,11] = 2
        #x1 = [pclass, sex, age, sibsp, parch, fare, embarked]
        # Number of features = 7                
        x = np.array([data[1:,2],data[1:,4],data[1:,5],data[1:,6],data[1:,7],data[1:,9],data[1:,11]])                   # feature vector - trial 1 (eliminating some features)
        x_y = np.array([x,y])
        return x_y


# Calling all functions

data = import_data()
x_y = data_preprocessing(data)
h_w = hypothesis(x_y[0])
j = cost(x_y[1],h_w[0])
w = train(j,h_w[1])
h_w2 = hypothesis(x_y[0],w)
prediction = predict(h_w2[0])
