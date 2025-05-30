import torch 
from tqdm import tqdm
import numpy as np 
import logging 
import matplotlib
import optuna

#################################################################################################################################################################
#                       CANN implementation using the invariants of the deformartion gradient prefviously calculatied from the Ogden model
#                       the principle stretches wil also be sampled randomnly and this time our helmholtz energy will come from the neural network
#                       the CANN will be trained based on the training data previously generated in the second task
#                                                                           RELu Rebels SS25
###################################################################################################################################################################

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