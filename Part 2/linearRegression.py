import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from matplotlib.pyplot import figure, plot, xlabel, ylabel, show, xlim, ylim

#loading the data
filename = 'data/ENB2012_data.csv'
df = pd.read_csv(filename)

#Extracting the attributes
raw_data = df.values
X = raw_data[:,:8]
attributeNames = np.asarray(df.columns[:8])

#Extracting the class labels (cooling load)
classLabels = raw_data[:,-1]

#Discretizing the cooling load
nbClassCL = 4
discrete_CL = []
for cl in classLabels:
    if cl <= 20:
        discrete_CL.append('1')
    elif cl <= 30:
        discrete_CL.append('2')
    elif cl <= 40:
        discrete_CL.append('3')
    else:
        discrete_CL.append('4')


#unique class labels
classNames = np.unique(discrete_CL)

#making a dictionary in order to assign each classlabel a unique number
classDict = {"1": 3, "2": 2, "3": 1, "4": 0}

#class index vector y 
y = np.array([classDict[cl] for cl in discrete_CL])

#N = data objects, M = attributes, C = number of classes
N, M = X.shape
C = len(classNames)

#Scaling the data (mean = 0, std = 1)
Y = StandardScaler().fit_transform(X)

#Each lambda creates a new model 
#lambdas = np.power(10.,range(-8,3)) 
lambdas2 = np.arange(0.0001,0.02,0.0001)
S = len(lambdas2)

# K-fold cross-validation for model selection
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initializing test error matrix
Error_test = np.zeros((S,K))

k=0
for train_index, test_index in CV.split(Y):

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]


    for i in range(S):

        ridge = Ridge(lambdas2[i])
        ridge.fit(X_train,y_train)
        

        # making predictions on test set
        y_pred_test = ridge.predict(X_test)
        Error_test[i,k] = mean_squared_error(y_test, y_pred_test)

 
        
    k+=1

print(Error_test)


#plotting the generalization error as a function on lambda:
f = figure()
plot(lambdas2, Error_test.mean(1))
xlabel('λ')
xlim(0,np.max(lambdas2))
ylim(np.min(Error_test.mean(1)),np.max(Error_test.mean(1)))
ylabel('Generalization error')
    
show()

print( "best lambda = ", lambdas2[np.argmin(Error_test.mean(1))] )
print( "min error = ", np.min(Error_test.mean(1)) )
