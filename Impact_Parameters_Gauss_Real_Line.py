#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 10:40:46 2025

@author: freulon
"""

#Import modules

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# In this script we compute the Gaussian entropic optimal transport for 
# a set of different regularization parameters. For illustration purposes,
# all the probability measures are one dimensional.

def cov_eot(var_a=1, var_b=1, eps=0, rho=0):
    """
    Optimal transport with penalty 2eps KL divergence.
    Reference covariance parameterized by its correlation parameter rho.
    Returns the covariance of the Gaussian coupling solution of the 
    entropic optimal transport between two centred 
    gaussian N(0,var_a) and N(0,var_b). 
    """
    m = 1+eps*rho/(np.sqrt(var_a*var_b)*(1-rho**2)) # computation of m.
    
    c = (np.sqrt(var_a*m*var_b*m + eps**2/4) - eps/2)/m # covariance solution

    cov_mat = np.array([[var_a, c],
                        [c, var_b]])
    return cov_mat

def cost_eot(var_a=1, var_b=1, eps=0, rho=0): 
    """
    Compute the entropic optimal transport cost with reference covariance
    with the input measures as marginals and correlation parameter rho.
    """
    m = 1+eps*rho/(np.sqrt(var_a*var_b)*(1-rho**2)) # computation of m
    root_ab_perturbed = np.sqrt(var_a*var_b*m**2 + eps**2/4)
    
    if eps==0:
        logdet=0
        constant=0
    else:
        logdet = eps*np.log(root_ab_perturbed + eps/2)
        constant = 2*eps/(1-rho**2) - eps-eps*np.log(eps) + eps*np.log(1-rho**2)
    
    eot_cost = var_a + var_b -2* root_ab_perturbed + logdet + constant
    return eot_cost

#Variance of the input distribution

var_a=1
var_b=1

mean = np.array([0,0]) #We only consider centered Gaussian measures.

#Building the grid for the evalutation of the density of the bivariate gaussian

x_grid = np.linspace(-2.2*var_a, 2.2*var_a, num=100)
y_grid = np.linspace(-2.2*var_b, 2.2*var_b, num=100)

X, Y = np.meshgrid(x_grid,y_grid)

#First set of experiment - classic entropic penalty : rho=0

eps_grid_small = [0.01, 0.1,0.5, 1, 2,  10]

#Computation of the pdf for the different values of the regularization parameters
pdf_list = []

for eps in eps_grid_small:
    cov_mat=cov_eot(var_a=var_a, var_b=var_b, eps=eps, rho=0) # Computation of the covariance matrix
    distr = multivariate_normal(cov = cov_mat, mean = mean)  # Instanciation of the 2d gaussian
    pdf = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pdf[i,j] = distr.pdf([X[i,j], Y[i,j]])
    pdf_list.append(pdf)

fig = plt.figure(figsize=(18,10))
for it, reg in enumerate(eps_grid_small):
    pdf_2d = pdf_list[it]
    plt.subplot(2,3,it+1)
    plt.contourf(X, Y, pdf_2d, cmap='viridis')
    plt.title(r'$\varepsilon=$'+str(reg)+r' and $\rho=0$', fontsize=20)

plt.savefig('EOT_2D_Gaussian_varepsilon.pdf', format='pdf')

#Second set of experiment - variation of the reference correlation

rho_grid_small = [0, 0.3, 0.6, 0.9, 0.95, 0.99]
rho_grid_full = np.linspace(0, 0.99, 100)

#Computation of the pdf for the different values of the regularization parameters
pdf_list = []

for rho in rho_grid_small:
    cov_mat=cov_eot(var_a=var_a, var_b=var_b, eps=2, rho=rho) # Computation of the covariance matrix
    distr = multivariate_normal(cov = cov_mat, mean = mean)  # Instanciation of the 2d gaussian
    pdf = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pdf[i,j] = distr.pdf([X[i,j], Y[i,j]])
    pdf_list.append(pdf)


fig = plt.figure(figsize=(18,10))
for it, rho in enumerate(rho_grid_small):
    pdf_2d = pdf_list[it]
    plt.subplot(2,3,it+1)
    plt.contourf(X, Y, pdf_2d, cmap='viridis')
    plt.title(r'$\varepsilon=2$ and $\rho=$'+str(rho), fontsize=20)

plt.savefig('EOT_2D_Gaussian_priorcov.pdf')

n_eps = len(eps_grid_small)
n_rho = len(rho_grid_full)

cost_excess = np.zeros((n_eps, n_rho))

for i in range(n_eps) :
    for j in range(n_rho):
        eps = eps_grid_small[i]
        rho = rho_grid_full[j]
        cost_excess[i,j] = cost_eot(var_a=var_a, var_b=var_b, eps=eps, rho=rho)    

cost_eot(var_a=var_a, var_b=var_b, eps=1, rho=0.9)


fig = plt.figure(figsize=(8,5))
for i in range(n_eps):
    eps = eps_grid_small[i]
    plt.plot(rho_grid_full, cost_excess[i], label = r'$\varepsilon=$'+str(eps))
    plt.legend(loc='best')
    plt.xlabel(r'$\rho$', fontsize=15)
    plt.ylabel(r'$W_{\Sigma}^{\varepsilon}-W_2^2$', fontsize=15)
    plt.xlim((-0.02, 1.01))

plt.savefig('Eot_curve_impact_correlation_ref.pdf')

# Excess of entropic optimal transport for varying epsilon

eps_grid_full = np.linspace(0, 10, 100)
n_eps_full = len(eps_grid_full)
n_rho_small = len(rho_grid_small)

cost_excess_eps = np.zeros((n_rho_small, n_eps_full))

for i in range(n_rho_small):
    for j in range(n_eps_full):
        eps = eps_grid_full[j]
        rho = rho_grid_small[i]
        cost_excess_eps[i,j] = cost_eot(var_a=var_a, var_b=var_b, 
                                        eps=eps, rho=rho)
        

fig = plt.figure(figsize=(8,5))
for i in range(n_rho_small):
    rho = rho_grid_small[i]
    plt.plot(eps_grid_full, cost_excess_eps[i], 
             label = r'$\rho=$'+str(rho))
    plt.legend(loc='best')
    plt.xlabel(r'$\varepsilon$', fontsize=15)
    plt.ylabel(r'$W_{\Sigma}^{\varepsilon}-W_2^2$', fontsize=15)

plt.savefig('Eot_curve_impact_epsilon_param.pdf')
 