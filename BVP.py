'''
This script implements the boundary value problem for computing the open-loop 
solution.
'''

import numpy as np
from scipy.integrate import solve_bvp
from scipy.integrate import solve_ivp
import time
import warnings

import NN_evaluation

def solve_BVP(problem, X0_samples, initialization=None, 
              init_config={'timesteps':30, 'timeseq':15, 'model':None}, 
              solver_config={'max_nodes':5000, 'tol':1e-5, 
                             'replace_sample':True}):
    '''Initial and progressive data generation for training, validation and 
    testing.
    
    Parameters
    ----------
    problem : OCP.ControlledReactionDiffusionSystem
        Optimal control problem.
    X0_samples : numpy.ndarray, shape (n, num_samples)
        Initial conditions.
    initialization : str, optional
        Initialization technique, either 'time_marching' or 'NN_warm_start'. 
        Default is basic initialization.
    init_config : dict of {'timesteps' : int, 'timeseq' : int or numpy.ndarray,
                           'model' : NN.Net}, optional
        Configuration of initialization technique.
    solver_config : dict of {'max_nodes' : int, 'tol' : float, 
                             'replace_sample' : bool}, optional
        Configuration of BVP solver.
        
    Returns
    -------
    t_data : numpy.ndarray, shape (num_data,)
        Time of data points. 
    X_aug_data : numpy.ndarray, shape (2n+1, num_data)
        State, Costate and value function of data points. 
    idx_samples : numpy.ndarray, shape (num_samples+1,)
        Indices of data points belonging to one trajectory. 
    X_aug_funs : list of scipy.interpolate.PPoly
        Solutions of BVP solver as C1 continuous cubic spline.
    '''
    # initialization of data storage
    t_data = []
    X_aug_data = np.empty((2*problem.n + 1,0))
    X_aug_funs = []
    
    # monitor data generation
    num_fails = 0
    fail_times = []
    num_solutions = 0
    solution_times = []
    total_time_solve_bvp = 0
    num_init_points = 0
    idx_samples = np.zeros(1)
    
    current_sample = 0    
    num_samples = X0_samples.shape[1]
    
    # BVP solver configuration
    max_nodes = solver_config['max_nodes']
    tol = solver_config['tol']
    replace_sample = solver_config['replace_sample']
        
    while current_sample < num_samples:
        # initial condition
        X0 = X0_samples[:, [current_sample]]

        # boundary condition
        def bc_X0(X_aug_t0, X_aug_tf):
            return problem.bc(X_aug_t0.reshape(-1,1), X_aug_tf.reshape(-1,1), 
                              X0).flatten()

        # track time
        start_time = time.time()

        try:
            if initialization == 'time_marching':
                timeseq = init_config['timeseq']
                if np.isscalar(timeseq): 
                    timeintervals = np.linspace(problem.t0, problem.tf, 
                                                timeseq)
                else:
                    timeintervals = problem.t0 + (problem.tf-problem.t0) \
                                    * timeseq
                num_timeintervals = len(timeintervals)
            
                t_mesh = np.array([problem.t0])
                X_aug = np.vstack((X0, np.zeros((problem.n+1, 1))))
                adaptive_tol = 1e-1

                for j in range(1, num_timeintervals):
                    # initialization for BVP solver
                    t_mesh = np.concatenate((t_mesh, timeintervals[[j]]))
                    X_aug = np.hstack((X_aug, X_aug[:,-1:]))
                    
                    # adapt tolerance
                    if j == num_timeintervals - 1:
                        adaptive_tol = tol
                    elif adaptive_tol >= 2.*tol:
                        adaptive_tol/=2.
                           
                    # application of BVP solver
                    start_time_solve_bvp = time.time()
                    sol = solve_bvp(problem.aug_dynamics, bc_X0, t_mesh, X_aug, 
                                    tol=adaptive_tol, max_nodes=max_nodes)
                    end_time_solve_bvp = time.time()

                    if not sol.success:
                        warnings.warn(sol.message, RuntimeWarning) 
                        raise RuntimeWarning

                    else:    
                        if j == num_timeintervals - 1:
                            num_init_points += len(t_mesh)
                        t_mesh = sol.x
                        X_aug = sol.y
                        X_aug_fun = sol.sol
            else:
                if initialization is None:
                    timesteps = init_config['timesteps']
                    # initialization for BVP solver
                    if np.isscalar(timesteps):
                        t_mesh = np.linspace(problem.t0, problem.tf, timesteps)
                    else:
                        t_mesh = problem.t0 + (problem.tf-problem.t0) \
                                * timesteps
                    X_aug = np.tile(np.vstack((X0, np.zeros((problem.n+1,1)))), 
                                    len(t_mesh))
                    
                elif initialization == 'NN_warm_start':
                    model = init_config['model']
                    # initialization for BVP solver
                    t_traj, X_traj = NN_evaluation.get_trajectory(problem, 
                                                                  model, X0)
                    V_traj, P_traj = NN_evaluation.get_data(model, t_traj,
                                                                  X_traj)
                    t_mesh = t_traj
                    X_aug = np.vstack((X_traj, P_traj, V_traj))
                
                # application of BVP solver
                start_time_solve_bvp = time.time()
                sol = solve_bvp(problem.aug_dynamics, bc_X0, t_mesh, X_aug, 
                                tol=tol, max_nodes=max_nodes)
                end_time_solve_bvp = time.time()

                if not sol.success:  
                    if num_fails >= 100:
                        raise RuntimeError('maximum number of fails is exceeded')
                    else:
                        warnings.warn(sol.message, RuntimeWarning) 
                        raise RuntimeWarning

                else:
                    num_init_points += len(t_mesh)
                    t_mesh = sol.x
                    X_aug = sol.y   
                    X_aug_fun = sol.sol

        except RuntimeWarning:
            # track time
            end_time = time.time()

            num_fails += 1
            fail_times.append(end_time - start_time)
                
            if replace_sample:
                # generate new sample
                X0_samples[:, [current_sample]] = np.random.uniform(
                    problem.X0_lb, problem.X0_ub, (problem.n, 1))
            else:
                current_sample +=1
                
        else:
            # track time
            end_time = time.time()

            # store data
            t_data = np.concatenate((t_data, t_mesh))
            X_aug_data = np.hstack((X_aug_data, X_aug))
            X_aug_funs.append(X_aug_fun)

            num_solutions +=1
            solution_times.append(end_time - start_time)
            total_time_solve_bvp += end_time_solve_bvp - start_time_solve_bvp
            idx_samples = np.hstack((idx_samples, idx_samples[-1] + len(t_mesh)))
            current_sample +=1
        
    print(f'number of fails:     {num_fails}')
    print(f'fail time:           {np.sum(fail_times):<7f}\n')
    print(f'number of solutions: {num_solutions} generating {len(t_data)} data points (from {num_init_points} initially guessed points)')
    print(f'solution time:       {np.sum(solution_times):<7f} of which {total_time_solve_bvp:7f} solving BVP')
    print(f'solution time per data point: {np.sum(solution_times)/len(t_data):<7f}\n')
        
    return t_data, X_aug_data, idx_samples, X_aug_funs
 
def get_trajectory(problem, X0, X_aug_fun, stoch_noise=[0.,0.], 
                   det_noise=[2.,3.,0.], ODE_solver='RK45', rtol=1e-3):
    '''Simulate trajectory (incl. noise) with BVP controller.'''
    # additional argument in dynamics (single time instance)
    def U_fun(t, X):
        assert np.isscalar(t)
        assert X.shape == (problem.n, 1)
        
        _, _, U = get_data(problem, np.array([t]), X_aug_fun)
        return U
    
    # integrate system with BVP controller
    sol = solve_ivp(problem.ODE_dynamics, [problem.t0, problem.tf], X0.flatten(), 
                    method=ODE_solver, vectorized=True, 
                    args=(U_fun, stoch_noise, det_noise), rtol=rtol)
        
    if not sol.success:
        warnings.warn(sol.message, RuntimeWarning) 
        raise RuntimeWarning
    
    t_traj = sol.t
    X_traj = sol.y
    
    return t_traj, X_traj

def get_data(problem, t, X_aug_fun):
    '''Compute value function, costate and control by evaluating continuous BVP
    solution at time instances.'''
    n = problem.n
        
    # apply continuous BVP solution
    X_aug = X_aug_fun(t)
        
    # split X_aug
    X = X_aug[:n] 
    V = X_aug[2*n:]
    P = X_aug[n:2*n]
    
    # compute control
    U = problem.U_optimal(X, P)
    return V, P, U