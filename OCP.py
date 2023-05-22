'''
This script contains the class implementing optimal control problems of 
reaction-diffusion systems.
'''

import numpy as np
import pickle
import os

# specific initial conditions and/or targets (should equal 0 on boundary)
def zero(Omega_mesh):
    return np.zeros(len(Omega_mesh))

def quadratic(Omega_mesh):
    return -(2*Omega_mesh - 1.)**2 + 1.

def polynomial(Omega_mesh):
    return -8*(Omega_mesh - 1.)**2 * Omega_mesh**2

def sin(Omega_mesh): 
    return np.sin(2*np.pi*Omega_mesh)

def quadratic_cos(Omega_mesh): 
    return -0.5*np.cos(4*np.pi*(Omega_mesh - 0.5)) * np.cos(2*np.pi*(Omega_mesh - 0.5)) - 0.5

class ControlledReactionDiffusionSystem():
    """1D finite difference disretized OCP of reaction-diffusion system.
    
    Parameters
    ----------
    n : int
        Number of discretization points.
    rxn : str, optional 
        Reaction term, either 'hyperbolic' or 'logistic'.
    ctrl : str, optional
        Control term, either 'add' or 'bilin'.
    ind_fun : tuple of (float, float), optional
        Support of indicator function in case of additive control term.
    target : function, optional
        Target. Default is zero.
    system_coeff : dict of {'a' : float, 'b' : float, 'c' : float}, optional
        Diffusion coefficient 'a', reaction coefficient 'b' and control 
        coefficient 'c'. Default is 1. for each system coefficient.
    t : tuple of (float, float), optional
        Initial and final time. Default is (0.,5.).
    Omega : tuple of (float, float), optional
        Spatial interval. Default is (0.,1.).
    X0 : tuple of (float, float), optional
        Initial condition domain. Default is (-1.5,1.5).
    """
    def __init__(self, n, rxn=None, ctrl=None, ind_fun=None, target=None, 
                 system_coeff={'a':1., 'b':1., 'c':1.},
                 cost_coeff={'alpha':1., 'beta':1., 'gamma':1.}, 
                 t=(0.,5.), Omega=(0.,1.), X0=(-1.5,1.5)):
        
        # time interval
        self.t = t
        self.t0 = t[0]
        self.tf = t[1]
        
        # 1D domain
        self.Omega = Omega
        self.Omega_lb = Omega[0]
        self.Omega_ub = Omega[1]
        
        # mesh 
        self.n = n
        self.mesh_size = 1/(n+1)
        self.Omega_mesh = np.linspace(self.Omega_lb, self.Omega_ub, n+2) 
        
        # initial condition domain
        self.X0 = X0
        self.X0_lb = np.full((n, 1), X0[0])
        self.X0_ub = np.full((n, 1), X0[1])
        
        # controlled reaction-diffusion system
        # diffusion coefficient
        self.a = system_coeff['a']        
        
        # reaction term with coefficient
        self.rxn = rxn          
        if rxn is not None:
            self.b = system_coeff['b']
        
        # source/sink term with coefficient
        self.ctrl = ctrl            
        if ctrl is not None:
            self.c = system_coeff['c']
        
        # discretized indicator function
        if ctrl == 'add':
            self.ind_fun = ind_fun
            if ind_fun is None:
                self.ind = np.ones((n,1))
            else:     
                self.ind = np.all(np.vstack((self.Omega_mesh[1:-1] >= ind_fun[0], 
                                            self.Omega_mesh[1:-1] <= ind_fun[1])), 
                                 axis=0).astype(int).reshape((n,1))
            
        # finite difference scheme for Laplace operator
        self.Q = self.a/self.mesh_size**2*(-2*np.eye(n) + np.eye(n, k=1) 
                                           + np.eye(n, k=-1))
            
        # coefficients of cost function
        self.alpha = cost_coeff['alpha']
        self.beta = cost_coeff['beta']
        self.gamma = cost_coeff['gamma']
        
        # target
        self.target = zero if target is None else target 
    
    def f(self, U, X):   
        '''Dynamics (multiple time instances).'''
        assert U.shape == (1, X.shape[1])

        # reaction term 
        if self.rxn is None:
            R = 0  
        elif self.rxn == 'hyperbolic':
            R = self.b*X**2 
        elif self.rxn == 'logistic':
            R = self.b*X*(1-X)

        # source/sink term
        if self.ctrl is None:
            S = 0 
        elif self.ctrl == 'add':
            S = self.c*U*self.ind
        elif self.ctrl == 'bilin':
            S = self.c*U*X
            
        return self.Q @ X + R + S

    def ODE_dynamics(self, t, X, U_fun, stoch_noise, det_noise):
        '''Dynamics of (noisy) ODE system (single time instance).'''
        assert np.isscalar(t)
        assert X.shape == (self.n, 1)
        
        # sample Gaussian noise (stochastic)
        mean, var = stoch_noise
        W = np.sqrt(var) * np.random.randn(self.n, 1) + mean
        # define shock noise (deterministic)
        start, end, factor = det_noise 
        W += factor * (t-start)**2*(t-end)**2 if (t>=start and t<=end) else 0.
        
        # evaluate control
        U = U_fun(t, X)          
        return self.f(U, X) + W

    def running_state_cost(self, X, Y):
        '''Running state cost incl. coefficient (multiple time instances).'''
        assert Y.shape == (X.shape[0], 1)
        return self.alpha*self.mesh_size*np.sum((X-Y)**2, axis=0, keepdims=True)
    
    def running_ctrl_cost(self, U):
        '''Running control cost incl. coefficient (multiple time instances).'''
        return self.beta*U**2

    def psi(self, U, X, Y):
        '''Total running cost (multiple time instances).'''
        assert U.shape == (1, X.shape[1])
        assert Y.shape == (X.shape[0], 1)
        return self.running_state_cost(X,Y) + self.running_ctrl_cost(U)
    
    def integrated_running_cost(self, t_mesh, running_cost):
        '''Integrated state, control or total running cost using composite 
        trapezoidal rule.'''
        assert running_cost.shape[1] == len(t_mesh)
        return np.trapz(running_cost.flatten(), x=t_mesh)

    def phi(self, X_tf, Y):
        '''Final cost.'''
        assert Y.shape == X_tf.shape
        return self.gamma*self.mesh_size*np.sum((X_tf-Y)**2)
    
    def J(self, t_mesh, U, X, Y):
        '''Objective/cost function.'''
        return self.integrated_running_cost(t_mesh, self.psi(U, X, Y)) + self.phi(X[:,[-1]], Y)
    
    def U_optimal(self, X, P):     
        '''Hamiltonian minimization condition: U as function of X and P.'''
        if self.ctrl is None:
            U = np.zeros((1, X.shape[1])) 
        elif self.ctrl == 'add':
            U = -self.c/(2*self.beta) * self.ind.T @ P
        elif self.ctrl == 'bilin':
            U = -self.c/(2*self.beta) * np.diag(P.T @ X).reshape((1,-1))                         
        return U

    def H_x(self, U, X, P, Y): 
        '''Adjoint equation: derivative of H w.r.t. x.'''
        assert X.shape == P.shape
        assert U.shape == (1, X.shape[1])
        assert Y.shape == (X.shape[0], 1)

        # derivative of reaction term
        if self.rxn is None:
            dRdx = 0
        elif self.rxn == 'hyperbolic':
            dRdx = 2*self.b*P*X 
        elif self.rxn == 'logistic':
            dRdx = self.b*P*(1-2*X)

        # derivative of source/sink term
        if self.ctrl is None or self.ctrl == 'add':
            dSdx = 0 
        elif self.ctrl == 'bilin':
            dSdx = self.c*U*P   
            
        return 2*self.alpha*self.mesh_size * (X-Y) + self.Q.T @ P + dRdx + dSdx
    
    def aug_dynamics(self, t_mesh, X_aug):
        '''Dynamics of BVP.'''
        X = X_aug[:self.n]
        P = X_aug[self.n:2*self.n]

        U = self.U_optimal(X, P)
        Y = self.target(self.Omega_mesh[1:-1]).reshape(-1,1)

        dXdt = self.f(U, X)
        dPdt = -self.H_x(U, X, P, Y)
        dVdt = -self.psi(U, X, Y)
        return np.vstack((dXdt, dPdt, dVdt))

    def bc(self, X_aug_t0, X_aug_tf, X0):
        '''Boundary conditions of BVP.'''
        X_t0 = X_aug_t0[:self.n]
        X_tf = X_aug_tf[:self.n]
        P_tf = X_aug_tf[self.n:2*self.n]
        V_tf = X_aug_tf[2*self.n:]

        Y = self.target(self.Omega_mesh[1:-1]).reshape(-1,1)

        # boundary condition for costate
        dphidx = 2*self.gamma*self.mesh_size*(X_tf-Y)

        bc_X = X_t0 - X0
        bc_P = P_tf - dphidx
        bc_V = V_tf - self.phi(X_tf, Y)
        return np.vstack((bc_X, bc_P, bc_V))    

    def save_config(self, problem_type):
        '''Save problem configurations.'''
        directory = get_directory(problem_type)            
        path = directory + '/problem_config.txt'
        with open(path, 'w') as f:
            f.write(f'------------------------------------\n {problem_type} Problem\n------------------------------------\n')
            f.write(f'dimension:           {self.n}\n')
            f.write(f'reaction:            {self.rxn}\n')
            f.write(f'control:             {self.ctrl}\n')
            if self.ctrl == 'add':
                f.write(f'indicator function:  {self.ind_fun}\n')
            f.write(f'target:              {self.target.__name__}\n')
            f.write(f'system coefficients: diffusion = {self.a}')
            if self.rxn is not None:
                f.write(f', reaction = {self.b}')
            if self.ctrl is not None:
                f.write(f', control = {self.c}')
            f.write('\n')
            f.write(f'cost coefficients:   running state = {self.alpha}, running control = {self.beta}, final = {self.gamma}\n')
            f.write(f't:                   {self.t}\n')
            f.write(f'Omega:               {self.Omega}\n')
            f.write(f'X0:                  {self.X0}\n')   
        
    def save_problem(self, problem_type):
        '''Save problem.'''
        directory = get_directory(problem_type)
        path = directory + '/problem.pickle'
        with open(path, 'wb') as f:
            pickle.dump(self, f) 

def load_problem(problem_type):
    '''Load problem.'''
    directory = get_directory(problem_type)
    path = directory + '/problem.pickle'
    with open(path, 'rb') as f:
        problem = pickle.load(f)
    return problem

def get_directory(problem_type):
    directory = 'experiments/' + problem_type + '_problem'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory