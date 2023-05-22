'''
This script implements the linear-quadratic regulator problem for computing the 
closed-loop solution for the linearized state dynamics.
'''

import numpy as np
from scipy.integrate import solve_ivp
import warnings

class LQProblem():
    """LQR problem of reaction-diffusion system with zero target.
    
    Parameters
    ----------
    problem : OCP.ControlledReactionDiffusionSystem
        Optimal control problem.
    """
    def __init__(self, problem):
        self.t0 = problem.t0
        self.tf = problem.tf
        self.n = problem.n
        self.ODE_dynamics = problem.ODE_dynamics
        
        # matrices of linear-quadratic problem
        # dynamics linearized around origin (dxdt ~= Fx + Gu)
        if problem.rxn is None or problem.rxn == 'hyperbolic':
            self.F = problem.Q
        elif problem.rxn == 'logistic':
            self.F = problem.Q + problem.b * np.eye(self.n)
        
        if problem.ctrl is None:
            self.G = np.zeros((problem.n,1))
        elif problem.ctrl == 'add':
            self.G = problem.c * problem.ind
        elif problem.ctrl == 'bilin':
            raise ValueError('bilinear control does not allow for LQR')
        
        # quadratic cost (J = int[x^TAx + u^TBu]dt + x^TCx)
        self.A = problem.alpha * problem.mesh_size * np.eye(problem.n) 
        self.B_inv = np.array([[1/problem.beta]])
        self.C = problem.gamma * problem.mesh_size * np.eye(problem.n)
        
        # save solution of RDE to avoid recomputation
        self.P_flat_fun = None
        
        # no transformation if target is zero
        self.transform_state = lambda X: X

    def solve_RDE(self, ODE_solver='RK45', rtol=1e-3):
        '''Solve Ricatti differential equation.'''
        # right-hand side of Ricatti differential equation (single time instance)
        def ODE_ricatti(t, P_flat):
            assert np.isscalar(t)
            assert P_flat.shape == self.F.flatten().shape
        
            P = P_flat.reshape(self.F.shape)
            PF =  P @ self.F
            dPdt = -PF - PF.T - self.A + P @ self.G @ self.B_inv @ self.G.T @ P
            return dPdt.flatten()
        
        sol = solve_ivp(ODE_ricatti, [self.tf, self.t0], self.C.flatten(),
                        dense_output=True, method=ODE_solver, rtol=rtol)
        self.P_flat_fun = sol.sol

    def get_trajectory(self, X0, stoch_noise=[0.,0.], det_noise=[2.,3.,0.], 
                       ODE_solver='RK45', rtol=1e-3):
        '''Simulate trajectory (incl. noise) with LQR controller.'''
        # additional argument in dynamics (single time instance)
        def U_fun(t, X):
            assert np.isscalar(t)
            assert X.shape == (self.n, 1)
            
            _, _, U = self.get_data(np.array([t]), X)
            return U
        
        # integrate closed-loop system with LQR controller
        sol = solve_ivp(self.ODE_dynamics, [self.t0, self.tf], X0.flatten(), 
                        method=ODE_solver, vectorized=True, 
                        args=(U_fun, stoch_noise, det_noise), rtol=rtol)
            
        if not sol.success:
            warnings.warn(sol.message, RuntimeWarning) 
            raise RuntimeWarning
        
        t_traj = sol.t
        X_traj = sol.y
        
        return t_traj, X_traj
    
    def get_data(self, t, X):
        '''Compute value function, costate and control (multiple time instances).'''
        assert t.shape == (X.shape[1],)
        assert X.shape[0] == self.n
    
        if self.P_flat_fun is None:
            self.solve_RDE()
            
        P_flat = self.P_flat_fun(t)
        P = P_flat.reshape(*self.F.shape, -1)
    
        # initialization of data storage
        V_data = np.empty((1,0))
        P_data = np.empty((self.n,0))
        U_data = np.empty((1,0))
        
        # iterate through time instances
        for j in range(t.shape[0]): 
            Z = self.transform_state(X[:,[j]])
            V_data = np.hstack((V_data, Z.T @ P[:,:,j] @ Z))
            P_data = np.hstack((P_data, (2 * P[:,:,j] @ Z)[:self.n]))
            U_data = np.hstack((U_data, -self.B_inv @ self.G.T @ P[:,:,j] @ Z))
            
        return V_data, P_data, U_data

class LQProblemNonzeroTarget(LQProblem):
    '''Augmented version for non-zero target.'''
    def __init__(self, problem):
        super(LQProblemNonzeroTarget, self).__init__(problem)
        
        # overwrite with augmented matrices
        zero_block = np.zeros((self.n, self.n))
        self.F = np.block([[self.F, self.F], [zero_block, zero_block]])
        self.G = np.block([[self.G], [np.zeros((self.n,1))]])
        self.A = np.block([[self.A, zero_block], [zero_block, zero_block]])
        self.C = np.block([[self.C, zero_block], [zero_block, zero_block]])
        
        # non-zero target
        self.Y = problem.target(problem.Omega_mesh[1:-1]).reshape(-1,1)
        
        # transformation to augmented state
        self.transform_state = lambda X: np.block([[X-self.Y], [self.Y]])