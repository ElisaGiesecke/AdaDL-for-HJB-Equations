# Solving high-dimensional Hamilton-Jacobi-Bellman Equations for Optimal Feedback Control via Adaptive Deep Learning Approach
This repository contains the code complementing my master thesis, available [here](https://github.com/ElisaGiesecke/AdaDL-for-HJB-Equations/blob/main/master%20thesis.pdf) with its defense [here](https://github.com/ElisaGiesecke/AdaDL-for-HJB-Equations/blob/main/master%20defense.pdf).

To demonstrate how Hamilton-Jacobi-Bellman Equations can be solved numerically via an adaptive deep learning approach, we provide a framework for computing optimal closed-loop controllers of reaction-diffusion systems. Specifically, we address an ODE-constrained optimal control problem arising from a spatial finite difference discretization, with the objective of steering the state towards a desired target. At the core of our algorithm lies a neural network modeling the value function and its gradient on which the the feedback control law is based. The network training relies on adaptively generated data obtained through the the open-loop solutions of corresponding boundary value problems.

Besides an empirical evaluation of the model accuracy, we compare the closed-loop solution obtained by the neural network model with the one resulting from the linear quadratic regulator approach. In simulations, we then observe the resulting feedback controls, as well as state and costate trajectories for a particular initial condition. Additionally, we consider the open-loop solution provided by the boundary value problem solver and the evolution of the uncontrolled system for the same initial condition and draw a comparison of the running, final and overall costs across all solutions. By introducing noise to the dynamical system, we further examine how well the controllers respond to disturbances. Lastly, we also pay special attention to the initialization for solving the boundary value problem in order to guarantee an efficient data generation in the pre-training, training and post-training phases. This numerical analysis is particularly conducted for high-dimensional systems in order to assess the scalability of our deep learning based solution method.


## Implementation
The code is written in Python (version 3.9.13) using NumPy (version 1.21.5) and Matplotlib (version 3.5.2), as well as the initial value problem and boundary value problem solvers of SciPy (version 1.9.1) and PyTorch (version 1.13.0) for an efficient implementation of neural networks.


## Usage
Following the modular programming approach, our functions and classes are grouped into separate `.py` files:

  * `OCP.py`: contains the class implementing optimal control problems of reaction-diffusion systems.
  * `BVP.py`: implements the boundary value problem for computing the open-loop solution.
  * `data_generation.py`: provides functionalities concerning the data generation.
  * `LQR.py`: implements the linear-quadratic regulator problem for computing the closed-loop solution for the linearized state dynamics.
  * `NN.py`: contains the classes implementing neural networks modeling the value function and its gradient.
  * `NN_optimization.py`: contains the class for training neural networks guided by heuristics.
  * `NN_evaluation.py`: evaluates the neural network for computing the closed-loop solution.
  * `simulation.py`: provides tools for the visualization of the simulations.

These files are executed by the enclosed Jupyter Notebooks. This interactive interface allows the user to gain an insight into the main steps of the algorithm and to flexibly choose the experiment setting, adjust hyperparameters and create simulations on-the-fly. Each notebook sets the configurations for three central components of the program, namely the problem, model and training. Additionally, the user can make key choices for the progressive and adaptive data generation, as well as for the evaluation of the final model. During an experimental run, the results are not only displayed within the notebook, but also saved externally along with the generated data. Moreover, the problem, model and training configurations and their respective class instances are stored offline, so that they can be reloaded for later usage. This [Figure](program_structure.pdf) portrays the building blocks of the program with the most important experiment choices (colored in green) and output files (colored in orange).


## Examples

We showcase selected numerical experiments via the provided `.ipynb` files: 

  * `exemplary set-up.ipynb`: provides a demonstration of the program's basic usage using a challenging high-dimensional optimal control problem as an example.
  * `linear-quadratic problem with noise.ipynb`: shows how the neural network controller reacts to disturbances in an additive control setting, as compared to the linear quadratic regulator and the boundary value problem solution.
  * `non-linear system with hyperbolic growth.ipynb` and `non-linear system with logistic growth.ipynb`: test the solution methods when a non-linear reaction term is introduced.
  * `training for bilinear control.ipynb`: demonstrates how the heuristical criteria, including early stopping, data set size selection and adaptive sampling, guide the training in a bilinear control setting.
  * `scalability for bilinear control 1.ipynb` and `scalability for bilinear control 2.ipynb`: examine the scalability of the adaptive deep learning method by empirically validating the model accuracy, simulating specific initial conditions and testing initializations for data generation.
  
The results of the experimental runs are stored in the `experiments` directory.
