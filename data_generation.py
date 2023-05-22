'''
This script provides functionalities concerning the data generation.
'''

import numpy as np
import torch
from torch.utils.data import Dataset

import OCP
import NN_optimization
import NN_evaluation

def sample_X0(problem, num_samples):
    '''Uniform sampling of initial conditions.'''
    return np.random.uniform(problem.X0_lb, problem.X0_ub, (problem.n, 
                                                            num_samples))
        
def select_X0(problem, model, X0_samples, num_top):
    '''Choose initial conditions with largest gradient norm for adaptive data 
    generation.'''
    num_samples = X0_samples.shape[1]
    t_samples = np.full(num_samples, fill_value=problem.t0)
    V_samples, P_samples = NN_evaluation.get_data(model, t_samples,
                                                  X0_samples)
    P_norm_samples = np.linalg.norm(P_samples, axis=0)
    top_idx = np.argpartition(P_norm_samples, -num_top)[-num_top:]
    return X0_samples[:, top_idx]   

def split_data(X_aug_data, get_U=False, problem=None):
    '''Split augmented data and compute control.''' 
    #n = problem.n
    n = int((X_aug_data.shape[0]-1)/2)
    
    # split X_aug
    X_data = X_aug_data[:n] 
    V_data = X_aug_data[2*n:]
    P_data = X_aug_data[n:2*n]
    
    # compute control
    if get_U:
        assert problem is not None
        U_data = problem.U_optimal(X_data, P_data)
        return X_data, V_data, P_data, U_data
    else:
        return X_data, V_data, P_data

def save_data(t_data, X_aug_data, idx_samples, problem_type, data_type):
    '''Save data to file.'''
    directory = OCP.get_directory(problem_type)
    path = directory + '/' + data_type + '_data.npz'
    np.savez(path, t=t_data, X_aug=X_aug_data, idx_samples=idx_samples)

def load_data(problem_type, data_type):
    '''Load data from file.'''
    directory = OCP.get_directory(problem_type)
    path = directory + '/' + data_type + '_data.npz'
    data = np.load(path)
    
    t_data = data['t']
    X_aug_data = data['X_aug']
    idx_samples = data['idx_samples']
    return t_data, X_aug_data, idx_samples

def update_data(t_data, X_aug_data, idx_samples, problem_type, data_type):
    '''Add new data to saved data.'''
    t_data_old, X_aug_data_old, idx_samples_old = load_data(problem_type, data_type)
    t_data = np.concatenate((t_data_old, t_data))
    X_aug_data = np.hstack((X_aug_data_old, X_aug_data))
    idx_samples = np.concatenate((idx_samples_old, idx_samples_old[-1] + idx_samples[1:]))
    save_data(t_data, X_aug_data, idx_samples, problem_type, data_type) 

class OCPDataset(Dataset):    
    '''Dataset for training, validation or testing.'''
    def __init__(self, problem_type, data_type):
        # load and split data
        t_data, X_aug_data, _ = load_data(problem_type, data_type)
        X_data, V_data, P_data = split_data(X_aug_data)
        
        # inputs
        self.t = torch.from_numpy(t_data).float().unsqueeze(0)
        self.X = torch.from_numpy(X_data).float()

        # outputs 
        self.V = torch.from_numpy(V_data).float()
        self.P = torch.from_numpy(P_data).float()

    def __len__(self):
        return self.X.size()[1]

    def __getitem__(self, idx):
        return self.t[:,idx], self.X[:,idx], self.V[:,idx], self.P[:,idx]
    
    def get_n(self):
        return self.X.size()[0]
    
def add_dataset_stats(problem_type, data_type, save=False, model_type=None, 
                      train_type=None):
    '''Display statistics of data sets.'''
    t_data, X_aug_data, idx_samples = load_data(problem_type, data_type)
    line = f'number of trajectories: {len(idx_samples) - 1}, number of data points: {len(t_data)}\n'
    
    print(line)
    
    if save:
        directory = NN_optimization.get_directory(problem_type, model_type, train_type)
        path = directory + '/statistics.txt'
        
        # append to existing file
        with open(path, 'a') as f:
            if data_type == 'train':
                f.write('training data set:\n')
            elif data_type == 'val':
                f.write('validation data set:\n')
            elif data_type == 'test':
                f.write('test data set:\n')
            f.write(line)