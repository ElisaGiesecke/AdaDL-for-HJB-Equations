'''
This script contains the classes implementing neural networks modeling the 
value function and its gradient.
'''

import torch
from torch import nn
import os
import warnings

class Net(nn.Module):
    '''Base class for all network models.'''
    def __init__(self):
        super(Net, self).__init__()
        self.depth = None
        self.width = None
        self.activation = None
        self.architecture = None
        self.forced_final_condition = None
        
        self.device = None
        
    def forward(self, t, X):
        '''Propagate data through the network.'''
        raise NotImplementedError
    
    def predict(self, t, X, mode, create_graph):
        '''Predict value function and its gradient.'''
        # allow gradient computation
        X.requires_grad=True
        X.grad = None
        
        if mode == 'eval':
            self.eval()
        elif mode == 'train':
            self.train()
        
        # forward propagation
        V_pred = self.forward(t, X)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')   
            # backward propagation
            V_pred.backward(gradient=torch.ones_like(V_pred), 
                            create_graph=create_graph)
        dVdx = X.grad
        X.detach_()
        return V_pred if create_graph else V_pred.detach(), dVdx
    
    def compute_ctrl(self, problem, X, dVdx):    
        '''Compute control.'''
        if problem.ctrl is None:
            U = torch.zeros(1 if X.dim()==1 else (X.size()[0],1))
        elif problem.ctrl == 'add':
            U = -problem.c/(2*problem.beta) * dVdx @ torch.from_numpy(problem.ind).float()
        elif problem.ctrl == 'bilin':
            U = -problem.c/(2*problem.beta) * torch.diag(dVdx @ X.T).reshape(-1,1)                         
        return U
    
    def save_config(self, problem_type, model_type, train_type):
        '''Save model configurations.'''
        directory = get_directory(problem_type, model_type, train_type)
        path = directory + '/model_config.txt'
        with open(path, 'w') as f:
            f.write(f'------------------------------------\n {model_type} Model\n------------------------------------\n')
            f.write(f'depth:                  {self.depth}\n')
            f.write(f'width:                  {self.width}\n')
            f.write(f'activation:             {self.activation}\n')
            f.write(f'architecture:           {self.architecture}\n')
            f.write(f'forced final condition: {self.forced_final_condition}\n')
    
    def save_model(self, problem_type, model_type, train_type):
        '''Save model.'''
        directory = get_directory(problem_type, model_type, train_type)
        path = directory + '/model.pth'
        torch.save(self, path)
      
def load_model(problem_type, model_type, train_type):
    '''Load model.'''
    directory = get_directory(problem_type, model_type, train_type)
    path = directory + '/model.pth'
    model = torch.load(path)
    return model 

def get_directory(problem_type, model_type, train_type, make=True):
    directory = 'experiments/' + problem_type + '_problem/' + train_type \
                + '_trained_' + model_type + '_model'
    if make and not os.path.exists(directory):
        os.makedirs(directory)
    return directory

class FeedforwardNet(Net):
    '''Feedforward network (same as NeuralNet with architecture='feedforward').'''
    def __init__(self, n, depth, width, activation, device):
        super(FeedforwardNet, self).__init__()
        
        self.depth = depth
        self.width = width
        self.activation = activation
        self.architecture = 'feedforward'
        self.forced_final_condition = 'no'
        
        self.device = device
        
        # input layer
        feedforward = [nn.Linear(n + 1, width), activation]  
        # hidden layers
        feedforward.extend(depth * [nn.Linear(width, width), activation])
        # output layer
        feedforward.append(nn.Linear(width, 1))
        self.feedforward = nn.Sequential(*feedforward)

    def forward(self, t, X):
        V = torch.cat((t,X), dim=-1)         
        return self.feedforward(V)

class NeuralNet(Net):
    '''Neural network (feedforward or residual).
    
    Parameters
    ----------
    n : int
        Number of discretization points.
    depth : int 
        Depth of network.
    width : int
        Width of network.
    activation : torch.nn.modules.activation
        Activation function of network, e.g. Tanh, ReLU, Softplus.
    architecture : str
        Network architecture, either 'feedforward' or 'residual'.
    device : str
        Device on which network is trained, either 'cpu' or 'cuda'.
    '''
    def __init__(self, n, depth, width, activation, architecture, device):
        super(NeuralNet, self).__init__()
        
        self.depth = depth
        self.width = width
        self.activation = activation
        self.archtitecture = architecture
        self.forced_final_condition = 'no'
        
        self.device = device
        
        # input layer
        self.input_layer = nn.Linear(n + 1, width)  
        
        # hidden layers
        hidden_layers = []
        for l in range(depth):
            hidden_layers.append(nn.Linear(width, width))
        self.hidden_layers = nn.ModuleList(hidden_layers)
        
        # output layer
        self.output_layer =nn.Linear(width, 1)
        
        # FeedforwardNet or ResNet
        self.architecture = architecture
        
    def forward(self, t, X):
        V = torch.cat((t,X), dim=-1)
        
        # input layer
        V = self.activation(self.input_layer(V))
                
        # hidden layers 
        if self.architecture == 'feedforward':
            for layer in self.hidden_layers:
                V = self.activation(layer(V))   
        elif self.architecture == 'residual':
            for layer in self.hidden_layers:
                V = V + self.activation(layer(V)) 
                   
        # output layer
        V = self.output_layer(V)
        
        return V

class ForcedFinalConditionNet(Net):
    '''Model enforcing final condition based on neural network.
    
    Parameters
    ----------
    model : NN.Net
        Neural network.
    problem : OCP.ControlledReactionDiffusionSystem
        Optimal control problem.   
    '''
    def __init__(self, model, problem):
        super(ForcedFinalConditionNet, self).__init__()
        self.model = model
        
        self.depth = model.depth
        self.width = model.width
        self.activation = model.activation
        self.archtitecture = model.architecture
        self.forced_final_condition = 'yes'
        
        self.device = model.device
        
        self.tf = problem.tf
        self.gamma = problem.gamma
        self.mesh_size = problem.mesh_size
        self.Y = problem.target(problem.Omega_mesh[1:-1])
        self.Y = torch.from_numpy(self.Y).float()
        
    def final_condition(self, X):
        return self.gamma*self.mesh_size*torch.sum((X-self.Y)**2, dim=-1, 
                                                   keepdim=True)
    
    def forward(self, t, X):
        tf = torch.full((1,) if X.dim()==1 else (X.size()[0],1), self.tf)
        return self.model(t, X) - self.model(tf, X) + self.final_condition(X)  