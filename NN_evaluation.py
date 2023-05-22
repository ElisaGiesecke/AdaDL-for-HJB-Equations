'''
This script evaluates the neural network for computing the closed-loop solution.
'''

import numpy as np
from scipy.integrate import solve_ivp
import torch
import warnings

def get_trajectory(problem, model, X0, stoch_noise=[0.,0.], 
                   det_noise=[2.,3.,0.], ODE_solver='RK45', rtol=1e-3): 
    '''Simulate trajectory (incl. noise) with NN controller.'''
    # additional argument in dynamics (single time instance)
    def U_fun(t, X):
        assert np.isscalar(t)
        assert X.shape == (problem.n, 1)
        
        _, _, U = get_data(model, np.array([t]), X, get_U=True, problem=problem)
        return U
    
    # integrate closed-loop system with NN controller
    sol = solve_ivp(problem.ODE_dynamics, [problem.t0, problem.tf], 
                    X0.flatten(), method=ODE_solver, vectorized=True, 
                    args=(U_fun, stoch_noise, det_noise), rtol=rtol)
        
    if not sol.success:
        warnings.warn(sol.message, RuntimeWarning) 
        raise RuntimeWarning
    
    t_traj = sol.t
    X_traj = sol.y
    
    return t_traj, X_traj

def get_data(model, t, X, get_U=False, problem=None):
    '''Evaluate NN (multiple time instances, numpy arrays).'''
    assert t.shape == (X.shape[1],)
     
    # transform numpy arrays into torch tensors   
    t_batch = torch.tensor(t).unsqueeze(1).float().to(model.device)
    X_batch = torch.tensor(X.T).float().to(model.device)
    
    V_pred_batch, dVdx_batch = model.predict(t_batch, X_batch, 'eval', False)
    
    # transform torch tensors into numpy arrays
    V = V_pred_batch.numpy().T
    P = dVdx_batch.numpy().T
    
    # compute control
    if get_U:
        assert problem is not None
        assert X.shape[0] == problem.n
        U = problem.U_optimal(X, P)
        return V, P, U
    else:
        return V, P