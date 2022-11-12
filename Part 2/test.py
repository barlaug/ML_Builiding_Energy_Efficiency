
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
import torch
from toolbox_02450 import train_neural_net

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

import warnings
warnings.filterwarnings('ignore')

datapath = "c:/Users/Amine/Documents/Amine/INSA/INSA_5IF_DTU/Machine_Learning/Project/ML_Energy_Efficiency/Part 2/data/"
filename = 'ENB2012_data.csv'
df = pd.read_csv(datapath+filename)
col_names = ['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height',
             'orientation', 'glazing_area', 'glazing_area_distribution', 'heating_load', 'cooling_load']
df.columns = col_names


# Set target and data
y = df['cooling_load']

X = df.iloc[:, :8]
X = pd.DataFrame(StandardScaler().fit_transform(X))
X.columns = col_names[:8]

K1, K2 = 2, 2
p_grid_reg = np.power(10., range(-7, 1))
p_grid_ann = [32, 64, 128, 256]

# Define the model structure
loss_fn = torch.nn.MSELoss()
max_iter = 10000
N, M = X.shape


n_hidden_units = 1024
X = torch.Tensor(X.values)
y = [[i] for i in y]
y = torch.Tensor(y)
print(X)
print(y)

dummy = DummyRegressor(strategy='mean')
dummy.fit(X, y)
print(mean_squared_error(dummy.predict(X), y))


ridge = Ridge(0.00001)
ridge.fit(X, y)
y_pred_test = ridge.predict(X)
print(mean_squared_error(y, y_pred_test))


def model(): return torch.nn.Sequential(
    # M features to H hiden units
    torch.nn.Linear(M, 512),
    # 1st transfer function, either Tanh or ReLU:
    torch.nn.Tanh(),  # torch.nn.ReLU(),
    # H hidden units to 1 output neuron
    torch.nn.Linear(512, 1)
)


net, final_loss, learning_curve = train_neural_net(model,
                                                   loss_fn,
                                                   X=X,
                                                   y=y,
                                                   n_replicates=1,
                                                   max_iter=max_iter)

y_test_pred_ann = net(torch.Tensor(X))
err_ann_inner = loss_fn(y, y_test_pred_ann)
print(err_ann_inner)


# res = {"outer_fold": [], "reg_lambda_i": [], "reg_test_error_i": [], "ANN_nb_i": [],
#        "ANN_test_error_i": [], "baseline_test_error_i": []}
# res["outer_fold"] = range(K1)

# # K-fold CrossValidation with two layers
# CV = model_selection.KFold(K1, shuffle=True)
# for k, (train_index, test_index) in enumerate(CV.split(X, y)):
#     X_train = X.iloc[train_index, :]
#     y_train = y.iloc[train_index]
#     X_test = X.iloc[test_index, :]
#     y_test = y.iloc[test_index]

#     ### Baseline estimation error ###
#     dummy = DummyRegressor(strategy='mean')
#     dummy.fit(X_train, y_train)
#     res['baseline_test_error_i'].append(
#         mean_squared_error(dummy.predict(X_test), y_test))

#     cv_inner = model_selection.KFold(K2, shuffle=True)

#     ### Ridge regression ###
#     ridge_model = RidgeCV(alphas=p_grid_reg, cv=cv_inner,
#                           scoring='neg_mean_absolute_error').fit(X_train, y_train)
#     best_param_reg = ridge_model.alpha_
#     y_pred_reg = ridge_model.predict(X_test)
#     err_reg = mean_squared_error(y_test, y_pred_reg)
#     res["reg_lambda_i"].append(best_param_reg)
#     res["reg_test_error_i"].append(err_reg)

#     ### ANN inner cross validation ###
#     # Initializing test error matrix
#     S = len(p_grid_ann)
#     Error_test = np.zeros((S, K2))
#     # K-fold cross-validation for model selection
#     for k, (train_index_inner, test_index_inner) in enumerate(cv_inner.split(X_train, y_train)):
#         print('\nCrossvalidation fold: {0}/{1}'.format(k+1, K2))

#         # Extract training and test set for current CV fold,
#         # and convert them to PyTorch tensors
#         X_train_inner = torch.Tensor(X_train.iloc[train_index_inner, :].values)
#         y_train_inner = torch.Tensor(y_train.iloc[train_index_inner].values)
#         X_test_inner = torch.Tensor(X_train.iloc[test_index_inner, :].values)
#         y_test_inner = torch.Tensor(y_train.iloc[test_index_inner].values)

#         # Compute the error for each number of hidden unit
#         for i, n_hidden_units in enumerate(p_grid_ann):

#             def model(): return torch.nn.Sequential(
#                 # M features to H hiden units
#                 torch.nn.Linear(M, n_hidden_units),
#                 # 1st transfer function, either Tanh or ReLU:
#                 torch.nn.Tanh(),  # torch.nn.ReLU(),
#                 # H hidden units to 1 output neuron
#                 torch.nn.Linear(n_hidden_units, 1)
#             )

#             net, final_loss, learning_curve = train_neural_net(model,
#                                                                loss_fn,
#                                                                X=X_train_inner,
#                                                                y=y_train_inner,
#                                                                n_replicates=1,
#                                                                max_iter=max_iter)

#             y_test_pred_ann = net(torch.Tensor(X_test_inner))
#             err_ann_inner = loss_fn(y_test_inner, y_test_pred_ann)
#             Error_test[i, k] = err_ann_inner

#     # Find the best number of hidden unit
#     generalization_error = Error_test.mean(1)
#     best_n_hidden_units = p_grid_ann[np.argmin(generalization_error)]

#     print(
#         f'\n\tBest loss error: {err_ann_inner} for {best_n_hidden_units} number of hidden units\n')

#     # Compute the final error
#     def model(): return torch.nn.Sequential(
#         # M features to H hiden units
#         torch.nn.Linear(M, best_n_hidden_units),
#         # 1st transfer function, either Tanh or ReLU:
#         torch.nn.Tanh(),  # torch.nn.ReLU(),
#         # H hidden units to 1 output neuron
#         torch.nn.Linear(best_n_hidden_units, 1)
#     )
#     net, final_loss, learning_curve = train_neural_net(model,
#                                                        loss_fn,
#                                                        X=torch.Tensor(
#                                                            X_train.values),
#                                                        y=torch.Tensor(
#                                                            y_train.values),
#                                                        n_replicates=1,
#                                                        max_iter=max_iter)
#     y_pred_ann = net(torch.Tensor(X_test.values))
#     err_ann = loss_fn(torch.Tensor(y_test.values), y_test_pred_ann)
#     print('\n\tBest loss final_loss: {}\n'.format(err_ann))

#     res["ANN_nb_i"].append(best_n_hidden_units)
#     res["ANN_test_error_i"].append(float(err_ann))

# print(pd.DataFrame.from_dict(data=res))
