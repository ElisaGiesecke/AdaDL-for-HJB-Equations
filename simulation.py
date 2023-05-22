'''
This script provides tools for the visualization of the simulations.
'''

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Palatino"
})
import os

def plot_X(t_mesh, Omega_mesh, X, ax=None):
    '''Plot state over time and space.'''
    Omega_grid, t_grid = np.meshgrid(Omega_mesh, t_mesh)
    X_incl_boundary = np.pad(X, ((1,1),(0,0)), mode='constant')
    if ax is None:
        ax = plt.axes(projection='3d')
    ax.plot_surface(Omega_grid, t_grid, X_incl_boundary.T, cmap='viridis')
    ax.set_xlabel('$\omega$')
    ax.set_ylabel('$t$')
    ax.set_zlabel('$X(t,\omega)$')
    
def plot_P(t_mesh, Omega_mesh, X, ax=None):
    '''Plot costate over time and space.'''
    Omega_grid, t_grid = np.meshgrid(Omega_mesh, t_mesh)
    X_incl_boundary = np.pad(X, ((1,1),(0,0)), mode='constant')
    if ax is None:
        ax = plt.axes(projection='3d')
    ax.plot_surface(Omega_grid, t_grid, X_incl_boundary.T, cmap='viridis')
    ax.set_xlabel('$\omega$')
    ax.set_ylabel('$t$')
    ax.set_zlabel('$P(t,\omega)$')
    
def plot_U(t_mesh, U, color=None, label=None):
    '''Plot control over time.'''
    plt.plot(t_mesh, U.flatten(), color=color, label=label, )
    plt.xlabel('$t$')
    plt.ylabel('$\mathbf{u}(t)$')  

def plot_V(t_mesh, V, color=None, label=None):
    '''Plot value function over time.'''
    plt.plot(t_mesh, V.flatten(), color=color, label=label)
    plt.xlabel('$t$')
    plt.ylabel('$V(t,X(t))$')

def plot_running_cost(problem, t_mesh, U, X, Y, color=None, label=None):
    '''Plot running cost resulting from state and control over time.'''
    plt.plot(t_mesh, problem.psi(U, X, Y).flatten(), linestyle='-', 
             color=color, label=label)
    plt.plot(t_mesh, problem.running_state_cost(X, Y).flatten(), 
             linestyle='--', color=color)
    plt.plot(t_mesh, problem.running_ctrl_cost(U).flatten(), linestyle=':', 
             color=color)
    plt.xlabel('$t$')
    plt.ylabel('$\psi(t,\mathbf{u}(t),X(t))$')

def display_costs(problem, t_mesh, U, X, Y, method, save=False, append=False, 
                  problem_type=None, model_type=None, train_type=None, 
                  init_type=None):
    '''Display costs with their components.'''
    lines = []
    lines.append(f'------------------------------------\n Costs of {method} Solution\n------------------------------------')
    lines.append(f'running state cost:   {problem.integrated_running_cost(t_mesh, problem.running_state_cost(X, Y)):<7f}')
    lines.append(f'running control cost: {problem.integrated_running_cost(t_mesh, problem.running_ctrl_cost(U)):<7f}')
    lines.append(f'total running cost:   {problem.integrated_running_cost(t_mesh, problem.psi(U, X, Y)):<7f}')
    lines.append(f'final cost:           {problem.phi(X[:,[-1]], Y):<7f}')
    lines.append(f'total cost:           {problem.J(t_mesh, U, X, Y):<7f}\n')
    
    for line in lines:
        print(line)
        
    if save:
        directory = get_directory(problem_type, model_type, train_type, init_type)
        path = directory + '/costs.txt'
        
        if append:
            with open(path, 'a') as f:
                for line in lines:
                    f.write(line + '\n')
        else:
            with open(path, 'w') as f:
                for line in lines:
                    f.write(line + '\n')
            
def get_directory(problem_type, model_type, train_type, init_type):
    directory = 'experiments/' + problem_type + '_problem/' + train_type \
                + '_trained_' + model_type + '_model/' + init_type \
                + '_initial_condition'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory