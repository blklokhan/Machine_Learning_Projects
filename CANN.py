import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np 
import logging 
import matplotlib
import optuna
from optuna.samplers import TPESampler

torch.set_default_dtype(torch.float32)
torch.set_num_threads(6)
device = torch.device("cpu")
device = torch.device(device)
model_file = "mymodel.torch"

#################################################################################################################################################################
#                       CANN implementation using the invariants of the deformartion gradient prefviously calculatied from the Ogden model
#                       the principle stretches wil also be sampled randomnly and this time our helmholtz energy will come from the neural network
#                       the CANN will be trained based on the training data previously generated in the second task
#                                                                           ReLU Rebels SS25
###################################################################################################################################################################
#Our objective is to design a CANN that a priori guarantees thermodynamic consistency of the function 
#We firstly define F for testing of the algorithm as we did in the 3rd task 

Lx = 2
Ly = 1
Nx = 80 * Lx + 1
Ny = 80 * Ly + 1
shape1 = (Nx, Ny)
dx = (Lx/(Nx-1))
dy = (Ly/(Ny-1))

#2D-Locations in the reference configuration of a body 
x_mesh, y_mesh = torch.meshgrid(torch.linspace(0, Lx, Nx), torch.linspace(0 , Ly, Ny),indexing='ij')
X = torch.stack([x_mesh.flatten(), y_mesh.flatten()], dim=1)

#non linerar diplacement field U 
N = X.shape[0]
F = torch.zeros(N, 2,2, dtype=torch.float32) #Deformation gradients

def deformation(X_p):
    u = torch.stack([2*X_p[0] + X_p[1], (0.8 * X_p[0])** 2])
    return X_p + u

for i in tqdm(range(N)):
    X_p = X[i].clone().detach().requires_grad_(True)
    F[i] = torch.autograd.functional.jacobian(deformation, X_p)

print(F)

#We will calculatre the invartiants of F and use them as inputs for our CANN
#We will sample lambda values randomly and train the CANN using the training data from the second task

#Hyperparameters of the NN and its training using TPE 

def objective(trial):
    #We define a learning rate 
    lr = trial.suggest_float("lr", 0.0001, 0.01 ) #log=True?
    layers = trial.suggest_int("layers", 2, 12)
    width = trial.suggest_int("width", 32, 512)
    #Criterion for the evaluation of the results 
    criterion = nn.MSELoss(reduction = "mean")

#training step will be defined here 

#We will create the neural netweork in this class

class MLNet(nn.Module):
    def __init__(self, input_dim=1, layers=2, width=64 ,output_dim=1):
        super(MLNet, self).__init__()
        self.hidden_dim = layers
        self.cann_1 = nn.Sequential(nn.Linear(input_dim, layers), nn.ReLU(), nn.Linear(layers, output_dim))
    
    def forward(self,x):
        out = self.cann_1(x)
        return out

