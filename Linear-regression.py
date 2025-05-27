#########################################################################################################################################################################################################
#                   PyTorch for Linear Regression.
#                   In this script we will Document and discuss what a simple regression in PyaTorch does mathematically. The results will be documented in a seperate file along with the introduction to Pytorch
#                   ReLU Rebels SS25
##########################################################################################################################################################################################################

import torch 
import numpy as np
import matplotlib.pyplot as plt

#Variables for Generating input data 
X = torch.arange(-5,5,0.1).view(-1,1)   #we want values from -5 to 5 in steps of 0.1
func = -5 * X                           #the underlying linear function that will help generate the input data 
Y = func + 0.4 * torch.randn(X.size())  #This is how we add noise to the data with Y being the output 
#                                        The addition of noise helps simulate real world data 

# Defining the function for forward pass for prediction
# This is the model being trained with 'w' defining our weights and 'b' defining the biases -  these are learnable parameters 
# See: Universal approximation theorem 
def forward(x):
    return w * x + b 

#Datapoint evalution with MSE(Mean Square Error)
#Our objective is to minimize the mean squared error loss function
# This will compare our Neural Network Outputs with the Target output fot the selected Data Tuples [NN Basics, Seite 11]
def criterion(y_pred, y):
    return torch.mean((y_pred -y) **2)

w = torch.tensor(-10.0, requires_grad=True)     #'requires_grad' has been set to true so that autograd records operatioins on the returned Tensor
b = torch.tensor(-20.0, requires_grad=True)     # see: https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor

step_size = 0.1
loss_list = []
iter = 20

for i in range(iter):

    #making predictions with forward pass
    Y_pred = forward(X)
    
    #Calculating the loss between original and predicted data points
    loss = criterion(Y_pred, Y)

    #The loss list was declarted to store the calculated losses in 
    loss_list.append(loss.item())

    #Backward pass for computing the gradients of the loss w.r.t to learnable parameters
    #This is the backpropagation step of our learning model: the graph is dofferentiatzed uding the chain rule, see: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html
    loss.backward()

    #After each iteration the parameters have to be updated
    #These parameters need to be updated as the weights and biases are learnable parameters  
    w.data = w.data - step_size * w.grad.data
    b.data = b.data - step_size * b.grad.data
    #After each iteration the gradients have to be brought back to zero, this prevents accumalation(according to pytorch documentation)
    w.grad.data.zero_()
    b.grad.data.zero_()

    #Printing valus for monitoring 
    #Its generally suggested to print outputs so that we know everything is going well :)
    #The variable i will iterate for the value set in the variable 'iter' (20 times for this case, so 0-19). 
    #The first column of the output is our calculated loss which is being added to the loss list with every iteration
    #The second column is our learned weights and the third column are the biases these will br put back into our 'Forward' function
    print('{}, \t{}, \t{}, \t{}'.format(i, loss.item(), w.item(), b.item()))

#Plotting the loss after each iteration 
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(loss_list, 'r')
plt.tight_layout()
plt.grid('True', color='y')
plt.xlabel("Epochs/Iterations")
plt.ylabel("Loss")

plt.subplot(2,1,2)
plt.plot(Y.detach(), 'rx')
plt.plot(func.detach(), 'k-')
plt.plot((w**X+b).detach(), 'c-')
plt.show()

##################################################################################################################################################################################################################################
#
#                           Linear Regression with a second function 
#                           We will implement the second function and also include the same noise function as we did in the first function 
#                           we will also convert the generated numpy data into PyTorch tensors
#                           The weight and biases stay the same as well as the minimization technique for our chosen loss function 
#                           
##################################################################################################################################################################################################################################

#this second function Y_2 is gonna create a new set of input data 

X2 = torch.linspace(-5, 5, 100).view(-1, 1)
func = -5 * X   
Y_2 = np.random.rand(100)*10
Y_2 = torch.tensor(Y_2, dtype=torch.float32).view(-1,1)

#New Loss list 
loss_list_2 = []

for j in range(iter):
    #same same but different
    Y2_pred = forward(X2)
    loss_2 = criterion(Y2_pred, Y_2)
    loss_list_2.append(loss_2.item())
    loss_2.backward()
    w.data = w.data - step_size * w.grad.data
    b.data = b.data - step_size * b.grad.data
    w.grad.data.zero_()
    b.grad.data.zero_()
    print('{}, \t{}, \t{}, \t{}'.format(j, loss_2.item(), w.item(), b.item()))


plt.subplot(3,1,1)
plt.plot(loss_list_2, 'r')
plt.tight_layout()
plt.grid('True', color='b')
plt.xlabel("Epochs/Iterations for numpy randomly generated data")
plt.ylabel("Loss for numpy randomly generated data")

plt.subplot(3,1,2)
plt.plot(Y_2.detach(), 'rx')
plt.plot(func.detach(), 'k-')
plt.plot((w**X+b).detach(),'c-')
plt.show()

#TO DO: FIX PLOTSSSS

