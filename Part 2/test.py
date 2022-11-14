
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
y = df['heating_load']

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
