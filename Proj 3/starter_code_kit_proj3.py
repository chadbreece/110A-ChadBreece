#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy.linalg import norm, solve, multi_dot
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


"""
@def   : objective function 
@param : x is vector 
@return: a scalar
"""
def objective_func(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

"""
@def   : gradient of objective function 
@param : x is vector 
@return: a vector 
"""
def grad_objective_func(x):
    return np.array([
        -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0]**2)      
    ])


def hessian_func(x):
    return np.matrix([
        [-400*(x[1] - 3*x[0]**2) + 2, -400 * x[0]], [-400 * x[0], 200]
    ])


# The stopping criteria is the gradient is less than 1e-9.


# In[2]:


# contour plot of objective function
# additional code: contour plots for objective function

# set the region to plot on
x = np.linspace(-0, 2, 500)
y = np.linspace(-2, 2, 500)
X, Y = np.meshgrid(x, y)

# evaluate function
Z = objective_func([X,Y])

# contour plots of objective function, with 15 contour lines.
plt.contour(X, Y, Z, 90);
# plt.contour(X, Y, Z, 15, colors='black'); # only black color


# In[3]:


"""
@def   : evaluation of model function
@param : p vector, g gradient vector, B model's matrix for quadratic term.
@return: a number
"""
def model(x, p):
    return objective_func(x) + np.dot(p, grad_objective_func(x)) + 0.5 * multi_dot([p.T, hessian_func(x), p])

"""
@def   : evaluate the actual reduction.
@param : x vector, p vector, g gradient vector, B model's matrix for quadratic term.
@return: a number
"""
def rho(x, p):
    return (objective_func(x) - objective_func(x + p)) / (model(x, np.array([0, 0]))  - model(x, p))

# framework for trust region method, we use the Hessian as the model's Bk.
"""
@def   : trust region method.
@param : subproblem_solver is a function, x0 starting point, radius0 the starting radius (also the max radius).
@return: a point
"""
def trust_region(subproblem_solver, x0, radius0):
    # The algorithm is modified from Algorithm 4.1 in the book.
    xcoords = [x0[0]]
    ycoords = [x0[1]]
    iter    = 0
    x = x0 
    radius = radius0
    err = norm(grad_objective_func(x))
    while (err > 1e-9):
        # main iterations for trust region.
        p = subproblem_solver(x, radius)
        # then test whether we should accept this point, calculate the ratio
        r = rho(x, p)
        if r < 0.25:
            # reject this because it is too small, x is not updated, only radius is updated
            radius = 0.5 * radius 
        elif r > 0.75:
            # accept this 
            radius = min(2 * radius , radius0) # cannot exceed maximum region size
            x = x + p
        else:
            # we also accept this, but region size is not changed
            x = x + p
            
        xcoords.append(x[0])
        ycoords.append(x[1])
        iter = iter + 1
            
        # update err
        err = norm(grad_objective_func(x))
            
    return x, iter, xcoords, ycoords


# In[4]:


"""
@def: Cauchy point method for the subproblem.
"""
def cauchy_point(x, radius):
    grad = grad_objective_func(x)
    hess = hessian_func(x)
    if multi_dot([grad.T, hess, grad]) <= 0:
        return -radius / norm(grad) * grad 
    else:
        k = norm(grad)**3 / (radius * multi_dot([grad.T, hess, grad]))
        return min(1, k) * (-radius / norm(grad) * grad)


# In[5]:


x0 = np.array([1.2, -1])
x, iter, xcoords, ycoords  = trust_region(cauchy_point, x0, 1)
print(x, iter)


# In[6]:


plt.contour(X, Y, Z, 90);
plt.plot(xcoords, ycoords, '-k', marker='.')

