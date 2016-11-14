import csv
import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

data_train = []
train = pd.read_csv("../data/pml_train.csv")
test = pd.read_csv("../data/pml_test_features.csv")
train.describe(include = [np.number])

"""
with open('../data/pml_train.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        data_train.append(row) 

data_train = np.array(data_train)
print np.shape(data_train)    
print data_train[0]
print data_train[1]


data_test = []
with open('../data/pml_test_features.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        data_test.append(row) 
data_test = np.array(data_test)



Xtrain = data_train[1:1000,-15:-1]
y_train = data_train[1:1000,-1]
Xtest = data_train[10000:10200,-15:-1]
y_test = data_train[10000:10200,-1]
#
# be carefully with dataset, make sure you select the correct feature
#Xtest = data_test[1:,-14:]
#print Xtrain[0]
#print Xtest[0]

Xtrain=np.array(Xtrain,dtype=float)
y_train=np.array(y_train,dtype=float)
Xtest=np.array(Xtest,dtype=float)
y_test = np.array(y_test,dtype=float)
#print Xtrain, y_train


model = SVR(C=1e3)
modelName = "SVM"



alpha = 0.2
#para0 = fit(Xtrain, y_train, alpha)
model.fit(Xtrain,y_train)
print("finish training")
# mean absolute loss (MAE) is used in the competition

#y_ = predict(para0, Xtrain)
y_ = model.predict(Xtrain)
print("finish prediction")
mse = ( (y_train - y_) ** 2).mean()
print('training mse:', mse)

mae = ( np.absolute(y_train - y_)).mean()
print('training mae:', mae)



#y_ = predict(para0, Xtest)
y_ = model.predict(Xtest)

num_test = len(y_) 
print num_test
mae = (np.absolute(y_test - y_)).mean()
print mae
#f = open('../data/testout.csv', 'wt')
#try:
#    writer = csv.writer(f)
#    writer.writerow( ('id', 'loss') )
#    for i in range(num_test):
#        writer.writerow( ( data_test[i+1, 0], y_[i] ) )
#finally:
#    f.close()
    
"""    
