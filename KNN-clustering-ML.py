# Author Dr. M. Alwarawrah
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import math, os, time, scipy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import (linear_model ,preprocessing,metrics)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# start recording time
t_initial = time.time()

#Columns names
col_names = ["region","tenure","age","marital","address","income","ed","employ","retire","gender","reside","custcat"]
#Read dataframe and skip first raw that contain header
df = pd.read_csv('teleCust1000t.csv',names=col_names, header = None, skiprows = 1)

#print Dataframe information
#print(df.describe())

#draw histograms for the following features
plt.clf()
df.hist()
plt.tight_layout()
plt.savefig("hist.png")

output_file = open('KNN_output.txt','w')

#define new data frame for all columns except 'custcat' and only take their values
X = df[["region","tenure","age","marital","address","income","ed","employ","retire","gender","reside"]] .values  

#define new data frame for 'custcat' values
Y = df['custcat'].values

#Normalizee data and find zero mean and unit variance:
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

#define train and test  and set the test size to 0.2 and random_state to 4
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=4)

# print the shape of each set:
print("X Train set dim: {} and Y Train set dim: {}".format(X_train.shape, Y_train.shape), file=output_file)
print("X Test set dim: {} and Y Test set dim: {}".format(X_test.shape, Y_test.shape), file=output_file)

#define empty lists
k_value_List= []
train_accuracy = []
test_accuracy = []
train_accuracy_std = []
test_accuracy_std = []
#loop over k value
for k in range(1,11):
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,Y_train)
    #prediction for test and train
    Y_test_predict = neigh.predict(X_test)
    Y_train_predict = neigh.predict(X_train)

    #append K value, Test accuracy, Train accuracy and corresponding standard deviation (STD) in the following lists
    k_value_List.append(k)
    train_accuracy.append(metrics.accuracy_score(Y_train, Y_train_predict))
    test_accuracy.append(metrics.accuracy_score(Y_test, Y_test_predict))
    test_accuracy_std.append(np.std(Y_test_predict==Y_test)/np.sqrt(Y_test_predict.shape[0]))
    train_accuracy_std.append(np.std(Y_train_predict==Y_train)/np.sqrt(Y_train_predict.shape[0]))
    
    #print K value and test & train accuracies 
    print("K value = %d, Train set Accuracy: %.2f +/- %.2f, Test set Accuracy: %.2f +/- %.2f"%(k, metrics.accuracy_score(Y_train, Y_train_predict), np.std(Y_train_predict==Y_train)/np.sqrt(Y_train_predict.shape[0]), metrics.accuracy_score(Y_test, Y_test_predict), np.std(Y_test_predict==Y_test)/np.sqrt(Y_test_predict.shape[0])), file=output_file)

#plot accuracy vs. K value for test and train data
plt.clf()
fig, ax = plt.subplots(1,2)
ax[0].errorbar(k_value_List, test_accuracy,yerr = test_accuracy_std,  color='k',ecolor='r', label = "Test Accuracy")
ax[0].set_xlabel("K value") 
ax[0].set_ylabel("Accuracy")
ax[0].legend(loc='best',frameon=False,fontsize = "8")
ax[1].errorbar(k_value_List, train_accuracy,yerr = train_accuracy_std,  color='k',ecolor='r', label = "Train Accuracy")
ax[1].set_xlabel("K value") 
ax[1].set_ylabel("Accuracy")
ax[1].legend(loc='best',frameon=False,fontsize = "8")
plt.tight_layout()
plt.savefig("accuracy_vs_k_value.png")

#print the best test accuracy with its K value 
print( "The best accuracy was with", max(test_accuracy), "with k=", test_accuracy.index(max(test_accuracy))+1, file=output_file)

output_file.close()
#End recording time
t_final = time.time()

t_elapsed = t_final - t_initial
hour = int(t_elapsed/(60.0*60.0))
minute = int(t_elapsed%(60.0*60.0)/(60.0))
second = t_elapsed%(60.0)
print("%d h: %d min: %f s"%(hour,minute,second))