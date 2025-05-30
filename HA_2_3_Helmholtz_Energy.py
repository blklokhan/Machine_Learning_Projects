import matplotlib
import torch
import numpy as np
import logging
from tqdm import tqdm

#torch settings
torch.set_default_dtype(torch.float32)
torch.set_num_threads(6)

#defining X
#The deformed configuration is given by x = X + u 
#Tensor u will be calculated from X, we wil apply a function on X
#We wil then use autograd to calculate the deformation gradient and derive the principal stretches from the deformation gradient
#finally the Helmholtz free energy for the constant material parameters will be implimented to get our PSI values for the CANN training 

#We will first define X as we did in the first homework task

Lx = 2
Ly = 1
Nx = 80 * Lx + 1
Ny = 80 * Ly + 1
shape1 = (Nx, Ny)
Nx = torch.tensor(Nx)
Ny = torch.tensor(Ny)
dx = (Lx/(Nx-1))
dy = (Ly/(Ny-1))

#2D-Locations in the reference configuration of a body 
x_mesh, y_mesh = torch.meshgrid(torch.linspace(0, Lx, Nx), torch.linspace(0 , Ly, Ny),indexing='ij')
X = torch.stack([x_mesh.flatten(), y_mesh.flatten()], dim=1)

#non linerar diplacement field U 
#u = torch.stack((2*X[:, 0] + X[:, 1], torch.pow(0.8*X[:, 0], 2)), dim=1)
#x = X + u 

N = X.shape[0]
F = torch.zeros(N, 2,2, dtype=torch.float32) #Deformation gradients

def deformation(X_p):
    u = torch.stack([2*X_p[0] + X_p[1], (0.8 * X_p[0])** 2])
    #u = torch.stack([
        #2 * X_p[0] + X_p[1] - X_p[2],(0.8 * X_p[0])**2 + 0.2 * X_p[2],torch.sin(X_p[1]) + 0.5 * X_p[2]**2])
    return X_p + u

for i in tqdm(range(N)):
    X_p = X[i].clone().detach().requires_grad_(True)
    F[i] = torch.autograd.functional.jacobian(deformation, X_p)

#Eigenvalues for helmholtz free energy
λ = torch.linalg.eigvals(F).real

#print("First 3 eigenvalues:\n", λ[:3])
#for i in range(N):
    #gradients = []
    #for j in range(2):
       # grad = torch.autograd.grad(outputs=x[i,j], inputs=X, grad_outputs=torch.tensor(1.0), retain_graph=True, create_graph=True)
        #gradients.append(grad[i])
    #F[i] = torch.stack(gradients, dim=1)
#print(F)

#material parameters for Helmholtz free energy
alpha = np.array([2.2519, -2.054, 10.01])
mu = np.array([2.2118, -0.061139, 3.9204e-7]) 
for i in range(10):
    λ1, λ2, = λ[i]
    λ3 = 1/λ1*λ2 
    #print(f"Point {i}: λ1 = {λ1}, λ2 = {λ2}, λ3 = {λ3}")
Hyper_elast = []
for i in range(len(alpha)):
    for j in range(len(mu)):
        psi = (mu[j]/alpha[i])*(λ1**alpha[i]+ λ2**alpha[i] + λ3**alpha[i]-3)
        #print(psi)
    Hyper_elast.append(psi.tolist())
print(Hyper_elast)

