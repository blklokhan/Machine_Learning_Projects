import torch
import matplotlib.pyplot as plt

#Torch Settings 

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)

plt.rcParams["text.usetex"] = True 
plt.rcParams["lines.markersize"] = 3
plt.rcParams["font.size"] = 18 

####################################################################################################################################################################

Lx = 2
Ly = 1 
Nx = 80 * Lx + 1 
Ny = 80 * Ly + 1

Shape1 = (Nx, Ny)

Nx = torch.tensor(Nx)
Ny = torch.tensor(Ny)

dx = (Lx / (Nx-1))
dy = (Ly / (Ny-1))

X = torch.meshgrid(torch.linspace(0, Lx, Nx), torch.linspace(0, Ly, Ny), indexing= 'ij')

X = torch.cat((X[0].reshape(-1, 1), X[1].reshape(-1, 1)), dim=1)

#################################################################################################
# Non-Linear displacement field  
#################################################################################################
#Example 
#u = torch.stack((2*X[:, 0] + X[:, 1], torch.pow(0.8*X[:, 0], 2)), dim=1)