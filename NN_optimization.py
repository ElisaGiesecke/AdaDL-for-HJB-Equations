'''
This script contains the class for training neural networks guided by 
heuristics.
'''

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Palatino"
})
import torch
from torch import nn
from torch.utils.data import DataLoader
import pickle
import os

class NetTraining():
    '''Training with progressive data generation, empirical validation and 
    testing of neural network model.
    
    Parameters
    ----------
    optimizer : torch.optim
        Optimization method, e.g. SGD, Adam, LBFGS.
    batch_size : int or str
        Batch size used by stochastic optimizer or 'full' for entire dataset 
        used by deterministic optimizer.
    grad_reg : float, optional
        Gradient regularization weight. Default is 0., i.e. no regularization.
    max_epoch : int, optional
        Maximum number of epochs for training termination. Default is 200.
    max_time : float, optional
        Maximum runtime  for training termination. Default is 1000..
    stop_params : dict of {'error_metric':[float, float, float], 'strip':int, 
                           'num_strips':int, 'max_iter':int, 'min_progress':float, 
                           'criterion':str or None, 'stop_tol':float}, optional
        Early stopping incl. performance measure (as linear combination of 
        error metrics), strip length, number of strips, maximum number of 
        iterations, minimum training progress, stopping criterion and tolerance.
    select_params : dict of {'subset_size':int or None, 'conv_tol':float, 
                             'growth_ub':float}, optional
        Data set size selection incl. subset size for variance approximation, 
        convergence tolerance, upper bound for data augmentation.
    '''
    def __init__(self, optimizer, batch_size, grad_reg=0., max_epoch=200, max_time=1000.,
                 stop_params={'error_metric':[0.,0.,1.], 'strip':5, 'num_strips':1, 'max_iter':50, 'min_progress':0.01, 'criterion':None, 'stop_tol':0.1},
                 select_params={'subset_size':None, 'conv_tol':1e-4, 'growth_ub':1.5}):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.grad_reg = grad_reg
        self.max_epoch = max_epoch
        self.max_time = max_time
        self.stop_params = stop_params
        self.select_params = select_params
    
        # store training, validation and test errors
        self.train_errors = np.empty((3,0))
        self.val_errors = np.empty((3,0))
        self.test_errors = np.empty((3,0))
        
        # monitor training
        self.current_epoch = 0
        self.opt_epoch = 0
        self.num_train_data = []
        self.stop_epochs = [0]
        self.train_times = []
        self.generation_ratios = []
    
    def loss(self, n, V, P, V_pred, dVdx):
        '''Compute loss.'''
        # initialize loss function
        loss_criterion = nn.MSELoss()
        
        value_loss = loss_criterion(V_pred, V)
        grad_loss = n * loss_criterion(dVdx, P)
        total_loss = value_loss + self.grad_reg * grad_loss
        return value_loss, grad_loss, total_loss
    
    def train(self, dataloader, model):
        '''Train neural network.'''
        dataset_size = len(dataloader.dataset)
        batch_size = dataloader.batch_size
        n = dataloader.dataset.get_n()
        
        # store loss over one epoch
        value_loss_epoch = []
        grad_loss_epoch = []
        total_loss_epoch = []
        
        # iterate through dataloader returning batches
        for batch, (t, X, V, P) in enumerate(dataloader):
            t, X, V, P = t.to(model.device), X.to(model.device), V.to(model.device), P.to(model.device)
            
            def closure():
                V_pred, dVdx = model.predict(t, X, 'train', True)
                
                value_loss, grad_loss, total_loss = self.loss(n, V, P, V_pred, dVdx)
                value_loss_epoch.append(value_loss)
                grad_loss_epoch.append(grad_loss)
                total_loss_epoch.append(total_loss)
        
                # backward propagation
                self.optimizer.zero_grad()
                total_loss.backward()
                return total_loss
                
            self.optimizer.step(closure)
                
            # print training progress
            if batch % 10 == 0:
                sample = batch * batch_size
                print(f'total loss: {total_loss_epoch[-1]:>7f}, value loss: {value_loss_epoch[-1]:>8f}, grad loss: {grad_loss_epoch[-1]:>8f} [{sample:>5d}/{dataset_size:>5d}]')
        
        return value_loss_epoch, grad_loss_epoch, total_loss_epoch

    def validate(self, dataloader, problem, model):
        '''Compute relative errors of value function, its gradient and control.'''
        num_batches = len(dataloader)
        
        # initialize error function
        error_criterion = nn.L1Loss()
       
        avg_value_RMAE = 0
        avg_grad_RML1 = 0
        avg_ctrl_RMAE = 0
        
        # iterate through dataloader returning batches
        for t, X, V, P in dataloader:
            t, X, V, P = t.to(model.device), X.to(model.device), V.to(model.device), P.to(model.device)
            
            V_pred, dVdx = model.predict(t, X, 'eval', False)
            U_pred = model.compute_ctrl(problem, X, dVdx)
            U = model.compute_ctrl(problem, X, P)
                     
            # compute relative errors
            avg_value_RMAE += error_criterion(V_pred, V)/error_criterion(torch.zeros_like(V), V)
            avg_grad_RML1 += error_criterion(dVdx, P)/error_criterion(torch.zeros_like(P), P)
            #avg_grad_RML2 += np.sum(np.sqrt(np.sum((dVdx.numpy() - P.numpy())**2, axis=1)))/np.sum(np.sqrt(np.sum(P.numpy()**2, axis=1)))
            avg_ctrl_RMAE += error_criterion(U_pred, U)/error_criterion(torch.zeros_like(U), U)
            
        avg_value_RMAE /= num_batches
        avg_grad_RML1 /= num_batches
        avg_ctrl_RMAE /= num_batches
        
        print(f'value RMAE: {avg_value_RMAE:<7f}, grad RML1: {avg_grad_RML1:<7f}, ctrl RMAE: {avg_ctrl_RMAE:<7f}')
        
        return np.array([avg_value_RMAE, avg_grad_RML1, avg_ctrl_RMAE]).reshape((-1,1))
    
    def early_stop(self):
        '''Update optimal epoch and check early stopping criterion.'''
        error_metric = np.array(self.stop_params['error_metric']).reshape((3, 1))
        val_error = np.sum(error_metric * self.val_errors,  axis=0)
        min_val_error = np.amin(val_error)
        self.opt_epoch = np.argmin(val_error) + 1
        generalization_loss = val_error[-1]/min_val_error - 1
        print(f'generalization loss: {generalization_loss:<7f}')
        
        # minimum required training epochs
        strip = self.stop_params['strip']
        num_strips = self.stop_params['num_strips']
        if self.train_errors.shape[1] <= strip * num_strips:
            return False
        
        train_error_strip = np.sum(error_metric * self.train_errors[:,-strip:],  axis=0)
        training_progress = np.sum(train_error_strip)/(strip * np.amin(train_error_strip)) - 1
        print(f'training progress:   {training_progress:<7f}')
        
        # maximum training epochs and minimum training progress
        max_iter = self.stop_params['max_iter']
        min_progress = self.stop_params['min_progress']
        if len(val_error) - self.stop_epochs[-1] >= max_iter or training_progress < min_progress:
            return True
        
        # stopping criteria
        criterion = self.stop_params['criterion']
        if criterion == 'GL':
            stop_tol = self.stop_params['stop_tol']
            stop = generalization_loss > stop_tol
        
        elif criterion == 'PQ':
            stop_tol = self.stop_params['stop_tol']
            stop = generalization_loss/training_progress > stop_tol
            
        elif criterion == 'UP':
            stop = True
            for j in range(num_strips - 1, -1, -1):
                if val_error[-1 - strip * j] <= val_error[-strip - 1 - strip * j]:
                    stop = False
                    break
        else:
            stop=False
                    
        return stop
    
    def select_dataset_size(self, dataset, model):   
        '''Dataset size selection incl. convergence test.'''
        # subset of dataset for approximation of variance and gradient
        dataset_size = len(dataset)
        subset_size = self.select_params['subset_size']
        if subset_size is None:
            subset_size = dataset_size
        n = dataset.get_n()
        
        # get number of trainable parameters
        num_params = sum(param.numel() for param in model.parameters() if 
                         param.requires_grad) 
        sample_grad = np.empty((0,num_params))
        
        # load single sample
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        for sample, (t, X, V, P) in enumerate(dataloader):
            t, X, V, P = t.to(model.device), X.to(model.device), V.to(model.device), P.to(model.device)
            V_pred, dVdx = model.predict(t, X, 'eval', True)
            _, _, total_loss = self.loss(n, V, P, V_pred, dVdx)
        
            # backward propagation
            model.zero_grad()
            total_loss.backward()
            
            # store loss gradients w.r.t. model parameters
            params_grad = []
            for param in model.parameters():
                params_grad.extend(param.grad.flatten().numpy())
            sample_grad = np.vstack((sample_grad, params_grad))
            
            if sample+1 == subset_size:
                break
        
        summed_sample_var = np.sum(np.var(sample_grad, axis=0))
        normed_sample_grad = np.linalg.norm(np.mean(sample_grad, axis=0), ord=1)
        
        # check convergence test
        conv_tol = self.select_params['conv_tol']
        growth_ub = self.select_params['growth_ub']
        if np.sqrt(summed_sample_var) <= conv_tol * normed_sample_grad \
           * np.sqrt(dataset_size):
            new_num_data = dataset_size
        else: 
            # dataset size selection with upper bound
            new_num_data = np.ceil(summed_sample_var/(conv_tol 
                                                     * normed_sample_grad)**2)
            new_num_data = int(min(growth_ub*dataset_size, new_num_data))
        return new_num_data   
    
    def plot_training_phase(self, save=False, problem_type=None, 
                               model_type=None, train_type=None):
        '''Plot training phase, i.e. relative errors, data set size and runtime 
        over epochs.'''
        num_epochs = self.train_errors.shape[1]
        labels = ['RMAE$_V$', 'RM$L^1_p$', 'RMAE$_u$']
        
        # create subplots on top of each other with shared x axis
        fig, axs = plt.subplots(4, sharex=True, constrained_layout=True)
        fig.suptitle('Training phase')
            
        # plot relative errors
        for i in range(3):
            axs[i].plot(np.arange(1,num_epochs + 1), self.train_errors[i], 
                        linestyle='--', color='blue', label='training error')
            axs[i].plot(np.arange(1,num_epochs + 1), self.val_errors[i], linestyle='-', 
                        color='blue', label='validation error')
            for stop_epoch in self.stop_epochs:
                line = axs[i].axvline(x=stop_epoch, linestyle=(0, (1, 3)), color='grey')
            line.set_label('data generation')
            axs[i].axvline(x=self.opt_epoch, linestyle=':', color='black', label='optimal epoch')
            axs[i].set_yscale('log')
            axs[i].set_ylabel(labels[i])
        axs[0].legend(bbox_to_anchor=(0, 1.05, 1, 0.2), loc='lower left', mode='expand', borderaxespad=0, ncol=2)
        
        # plot number of training data
        axs[-1].plot(np.arange(1,len(self.num_train_data) + 1), self.num_train_data, color='green')
        axs[-1].axvline(x=self.opt_epoch, linestyle=':', color='black')
        for stop_epoch in self.stop_epochs:
            axs[-1].axvline(x=stop_epoch, linestyle=(0, (1, 3)), color='grey')
        axs[-1].set_ylabel('$|D_{train}|$')
        axs[-1].tick_params(axis='y', colors='green')
        axs[-1].set_xlabel('epochs')
        
        # plot training time
        ax = axs[-1].twinx()
        ax.plot(np.arange(1,len(self.train_times) + 1), np.cumsum(self.train_times), color='red')
        ax.set_ylabel('runtime (s)')
        ax.tick_params(axis='y', colors='red')
        ax.spines['left'].set_color('green')
        ax.spines['right'].set_color('red')
        
        # save plot
        if save:
            directory = get_directory(problem_type, model_type, train_type)
            path = directory + '/training_phase.png'
            plt.savefig(path)
        plt.show()

    def display_stats(self, save=False, problem_type=None, model_type=None, 
                      train_type=None):
        '''Display training statistics.'''
        lines = []
        lines.append('------------------------------------\n Training Statistics\n------------------------------------')
        lines.append(f'total training time:           {sum(self.train_times):<7f}')
        lines.append(f'total number of training data: {self.num_train_data[-1]}\n')
        lines.append('data generation occured at epochs ' + ', '.join(f'{epoch}' for epoch in self.stop_epochs[1:-1]) + ' with ratio ' + ', '.join(f'{ratio:<3f}' for ratio in self.generation_ratios[:-1]))
        lines.append(f'total epochs: {self.stop_epochs[-1]}, optimal epoch: {self.opt_epoch}\n')
        lines.append('training errors of optimal model:')
        lines.append(f'value RMAE: {self.train_errors[0, self.opt_epoch-1]:<7f}, grad RML1: {self.train_errors[1, self.opt_epoch-1]:<7f}, ctrl RMAE: {self.train_errors[2, self.opt_epoch-1]:<7f}')
        lines.append('validation errors of optimal model:')
        lines.append(f'value RMAE: {self.val_errors[0, self.opt_epoch-1]:<7f}, grad RML1: {self.val_errors[1, self.opt_epoch-1]:<7f}, ctrl RMAE: {self.val_errors[2, self.opt_epoch-1]:<7f}\n')
        
        for line in lines:
            print(line)
            
        if save:
            directory = get_directory(problem_type, model_type, train_type)
            path = directory + '/statistics.txt'
            with open(path, 'w') as f:
                for line in lines:
                    f.write(line + '\n')

    def add_test_stats(self, problem_type, model_type, train_type):
        '''Display test statistics.'''
        directory = get_directory(problem_type, model_type, train_type)
        path = directory + '/statistics.txt'
        
        # append to existing file
        with open(path, 'a') as f:
            f.write('------------------------------------\n Test Statistics\n------------------------------------\n')
            f.write('test errors of optimal model:\n')
            f.write(f'value RMAE: {self.test_errors[0,-1]:<7f}, grad RML1: {self.test_errors[1,-1]:<7f}, ctrl RMAE: {self.test_errors[2,-1]:<7f}\n\n')

    def save_config(self, problem_type, model_type, train_type):
        '''Save training configurations.'''
        directory = get_directory(problem_type, model_type, train_type)
        path = directory + '/training_config.txt'
        with open(path, 'w') as f:
            f.write(f'------------------------------------\n {train_type} Training\n------------------------------------\n')
            f.write(f'optimizer:                      {self.optimizer}\n')
            f.write(f'batch size:                     {self.batch_size}\n')
            f.write(f'gradient regularization weight: {self.grad_reg}\n')
            f.write(f'maximum epochs:                 {self.max_epoch}\n')
            f.write(f'maximum time:                   {self.max_time}\n')
            error_metric = self.stop_params['error_metric']
            strip = self.stop_params['strip']
            num_strips = self.stop_params['num_strips']
            max_iter = self.stop_params['max_iter']
            min_progress = self.stop_params['min_progress']
            criterion = self.stop_params['criterion']
            stop_tol = self.stop_params['stop_tol']
            f.write(f'early stopping:                 error metric = {error_metric}, strip = {strip}, number of strips = {num_strips}, maximum iterations = {max_iter}, minimum progress = {min_progress}, criterion = {criterion}, tolerance = {stop_tol}\n')
            conv_tol = self.select_params['conv_tol']
            growth_ub = self.select_params['growth_ub']
            f.write(f'dataset size selection:         convergence tolerance = {conv_tol}, growth upper bound = {growth_ub}\n') 

    def save_training(self, problem_type, model_type, train_type):
        '''Save training.'''
        directory = get_directory(problem_type, model_type, train_type)
        path = directory + '/training.pickle'
        with open(path, 'wb') as f:
            pickle.dump(self, f) 
    
def load_training(problem_type, model_type, train_type):
    '''Load training.'''
    directory = get_directory(problem_type, model_type, train_type)
    path = directory + '/training.pickle'
    with open(path, 'rb') as f:
        training = pickle.load(f)
    return training

def get_directory(problem_type, model_type, train_type):
    directory = 'experiments/' + problem_type + '_problem/' + train_type \
                + '_trained_' + model_type + '_model'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory