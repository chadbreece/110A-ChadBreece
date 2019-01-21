# This starter code kit is written in Python and solves a sample problem with 
# the exact steepest descent method (gradient descent method). 
# 
# 1. The objective function is f(x) = x[0]**2 + x[1]**4
# 2. The optimal solution is (0, 0) obviously.

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

"""
@def   : objective function 
@param : x is vector 
@return: a scalar
"""
def objective_func(x):
    return x[0]**2 + x[1]**4

"""
@def   : gradient of objective function 
@param : x is vector 
@return: a vector 
"""
def grad_objective_func(x):
    return np.array([2*x[0], 4*x[1]**3])

# The exact steepest descent method. 
# The step length is chosen to solve a one-dimensional optimization problem
# The stopping criteria is selected to be 1e-9 for the gradient norm.
# The starting point is [5, 5] for this problem
# The optimal solution is [0., 0.].

x0 = np.array([5, 5])
x_opt = np.array([0., 0.])

"""
@def   : exact steepest descent method
@param : x is vector 
@return: a vector 
"""
def exact_steepest_descent(x0):             # input is the starting point
    x = x0                                  # select the starting point
    p = -grad_objective_func(x)             # find descent direction
    while norm(p) > 1e-9:                   # if the norm is not small
        def subproblem1d(alpha):            # define a 1d subproblem 
            return objective_func(x + alpha * p)  
                                            # use minimize_scalar function
        res = minimize_scalar(subproblem1d) # to solve the subproblem
        x = x + res.x * p                   # locate the next iterate
        p = -grad_objective_func(x)         # find next descent direction

    return x


x = exact_steepest_descent(x0)
print(x)
