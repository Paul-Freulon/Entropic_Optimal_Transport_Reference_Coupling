#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:06:35 2026

@author: freulon
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from scipy.linalg import sqrtm
from scipy.linalg import expm
from scipy.linalg import pinv
from numpy.linalg import eigh

K = np.array([[3,0],[2,3]]) # Drift matrix
L = np.eye(2)               # Diffusion matrix
init_var= 10*np.eye(2)  
t=0.2

def integrate_expmat_OU(t, K, L):
    """
    K is the drift matrix and L is the diffusion matrix.
    The SDE is dX_t = -K X_t dt + L dW_t. To compute Var(X_t),
    we need to compute the integral int_0^t exp(s K)LL^T exp(sK^T)ds.
    """
    t_grid = np.linspace(0, t, num=max(int(np.ceil(t*2000)),10))
    delta = t_grid[1]-t_grid[0]
    grid_exp_mat = [expm(s*K)@L for s in t_grid]
    grid_to_sum = [M@np.transpose(M) for M in grid_exp_mat]
    output = delta*np.sum(grid_to_sum,axis=0)
    return output

#I_mat= integrate_expmat_OU(t=0.1, K=K, L=L)
#print(I_mat)

def var_mat_OU(t, K, L, A_0=np.eye(2)):
    """
    K is the drift matrix and L is the diffusion matrix.
    The SDE is dX_t = -K X_t dt + L dW_t with X_0 sim N(A_0).
    The output is the variance matrix of X_t.
    """
    I_t = integrate_expmat_OU(t=t, K=K, L=L)
    exp_drift = expm(-t * K)
    output = exp_drift @ (A_0 + I_t) @ np.transpose(exp_drift)
    return output

A_t = var_mat_OU(t=1, K=K, L=L)
print('Marginal Variance')
print(A_t)
print('Its eigen values')
print(eigh(A_t)[0])

def cross_cov_OU(t_0, t_1, K, L, A_0=np.eye(2)):
    """
    K is the drift matrix, L is the diffusion matrix, t_0 <= t_1.
    The SDE is dX_t = -K X_t dt + L dW_t with X_0 sim N(A_0).
    The output is the cross covariance between X_{t_0} and X_{t_1}.
    That is E(X_{t_0}X_{t_1}^T).
    """
    var_mat = var_mat_OU(t=t_0, K=K, 
                         L=L, A_0=A_0)
    right_mat = expm(-(t_1 - t_0)*np.transpose(K))

    cross_cov = var_mat @ right_mat
    return cross_cov

print('Cross covariance')
print(cross_cov_OU(t_0=0.2, t_1=1, K=K, L=L))

def couple_cov_OU(t_0, t_1, K, L, A_0=np.eye(2)):
    """
    K is the drift matrix, L is the diffusion matrix, t_0 <= t_1.
    The SDE is dX_t = -K X_t dt + L dW_t with X_0 sim N(A_0).
    The output is the covariance of the couple (X_{t_0}, X_{t_1}).
    """
    var_0 = var_mat_OU(t=t_0, K=K, L=L, A_0=A_0)
    var_1 = var_mat_OU(t=t_1,  K=K, L=L, A_0=A_0)
    cov_01 = cross_cov_OU(t_0=t_0, t_1=t_1, K=K, L=L, A_0=A_0)
    couple_cov = np.block([[var_0, cov_01],
                        [np.transpose(cov_01),var_1]])
    return couple_cov

print('Covariance of the coupling')
cpl_cov = couple_cov_OU(t_0=0.1, t_1=1, K=K, L=L)
print(cpl_cov)
print('Its eigen values')
print(eigh(cpl_cov)[0])


def grid_margevar_OU(t_grid, K, L, A_0=np.eye(2)):
    """
    K is the drift matrix, L is the diffusion matrix, t_0 <= t_1.
    The SDE is dX_t = -K X_t dt + L dW_t with X_0 sim N(A_0).
    Output the list of covariance matrices V[X_t] at all time t in t_grid.
    """
    marginal_storage = []
    for t in t_grid:
        var_marg = var_mat_OU(t=t, K=K, L=L, A_0=A_0)
        marginal_storage.append(var_marg)
    return marginal_storage

print('List of time-marginal covariances')
grid_marge = grid_margevar_OU(t_grid=np.linspace(0,1,10), K=K, L=L)
for var in grid_marge:
    print(var)

def grid_cplcov_OU(t_grid, K, L, A_0=np.eye(2)):
    """
    K is the drift matrix, L is the diffusion matrix, t_0 <= t_1.
    The SDE is dX_t = -K X_t dt + L dW_t with X_0 sim N(A_0).
    Output the list of covariance matrices of the couples (X_{t_j}, X_{t_{j+1}})
    at all time t_j in t_grid.
    """
    cpl_cov_storage = []
    n = len(t_grid)
    for j in range(n-1):
        t_0 = t_grid[j]
        t_1 = t_grid[j+1]
        cpl_cov = couple_cov_OU(t_0=t_0, t_1=t_1, K=K, L=L, A_0=A_0)
        cpl_cov_storage.append(cpl_cov)
    return cpl_cov_storage

print('List of coupling covariances')
grid_cpl = grid_cplcov_OU(t_grid=np.linspace(0,1,10), K=K, L=L)
for cpl_cov in grid_cpl:
    print(cpl_cov)

def kernel_fBm(t_min, t_max, H=1/2):
    """
    Scalar kernel of a rescaled fractional Brownian motion.
    It is rescaled in order to have constant marginal variance equal to one.
    """
    temp = (t_max+1)**(2*H) + (t_min+1)**(2*H) - (t_max-t_min)**(2*H)
    const = 2 * (t_max+1)**(2*H)
    return temp/const

def couple_cov_fBm(t_min, t_max, H=1/2):
    """
    Coupling covariance of a 2 dimensional fractional Brownian motion
    with independent coordinates.
    """
    rho = kernel_fBm(t_min=t_min, t_max=t_max, H=H)
    couple_cov = np.block([[np.eye(2), rho*np.eye((2))],
                        [rho*np.eye(2), np.eye(2)]])
    return couple_cov


def grid_ref_cov_fBm(t_grid, H=1/2):
    """
    Coupling reference computed at every time of the grid. The reference
    dynamic is a fractional brownian motion rescaled in order to have 
    constant variance equal to one.
    """
    ref_cov_storage = []
    n = len(t_grid)
    for j in range(n-1):
        t_0 = t_grid[j]
        t_1 = t_grid[j+1]
        ref_cov = couple_cov_fBm(t_min=t_0, t_max=t_1, H=H)
        ref_cov_storage.append(ref_cov)
    return ref_cov_storage

def kernel_heat(t_min, t_max, sigma=1):
    """
    Scalar heat kernel rho(t,s)=exp(-(t-s)^2/(2*sigma^2)).
    """
    sq_dist = (t_max-t_min)**2
    return np.exp(-sq_dist/(2*sigma**2))

def couple_cov_heat(t_min, t_max, sigma=1):
    rho = kernel_heat(t_min=t_min, t_max=t_max, sigma=sigma)
    couple_cov = np.block([[np.eye(2), rho*np.eye((2))],
                        [rho*np.eye(2), np.eye(2)]])
    return couple_cov
    

def grid_ref_cov_heat(t_grid, sigma=1):
    """
    Coupling reference computed at every time of the grid. The reference
    dynamic is defined by the heat kernel with independ coordinates.
    """
    ref_cov_storage = []
    n = len(t_grid)
    for i in range(n-1):
        t_0 = t_grid[i]
        t_1 = t_grid[i+1]
        ref_cov = couple_cov_heat(t_min=t_0, t_max=t_1, sigma=sigma)
        ref_cov_storage.append(ref_cov)
    return ref_cov_storage

cplcov_grid_heat = grid_ref_cov_heat(np.linspace(0,1, num=7))
print(len(cplcov_grid_heat))

def couple_cov_mix(t_min, t_max, H=0.25, sigma=1):
    rho_fbm = kernel_fBm(t_min=t_min, t_max=t_max,H=H)
    rho_heat = kernel_heat(t_min=t_min, t_max=t_max, sigma=sigma)
    rho_mat = np.diag((rho_fbm, rho_heat))
    couple_cov = np.block([[np.eye(2), rho_mat],
                        [rho_mat, np.eye(2)]])
    return couple_cov

print('Coupling with mixed regularity')

def grid_ref_covmix(t_grid, H=0.25,  sigma=1):
    """
    Coupling reference computed at every time of the grid. The reference
    dynamic is defined by the fBM kernel for the first coordinates, and by
    a heat kernel for the second coordinate.
    """
    ref_cov_storage = []
    n = len(t_grid)
    for i in range(n-1):
        t_0 = t_grid[i]
        t_1 = t_grid[i+1]
        ref_cov = couple_cov_mix(t_min=t_0, t_max=t_1, H=H, sigma=sigma)
        ref_cov_storage.append(ref_cov)
    return ref_cov_storage

def sample_given_marginal(sample_X, couple_cov):
    """
    Given Gaussian X sim N(A) and block coupling covariance (A & C\\ C^T & B),
    output Y sim N(B) such that the couple (X, Y) has 
    distribution N((A & C\\ C^T & B)).
    The number of output samples from N(B) equals the number of input samples 
    from N(A).
    """
    n_sample = sample_X.shape[0]
    
    A = couple_cov[:2, :2]
    B = couple_cov[2:, 2:]
    C = couple_cov[:2, 2:]
    schur = B - np.transpose(C) @ pinv(A) @ C

    indep_sample = multivariate_normal(mean=np.zeros(2), cov=schur, size=n_sample)
    transform_first_marg = pinv(A) @ np.transpose(sample_X)
    transform_first_marg = np.transpose(C) @ transform_first_marg 
    transform_first_marg = np.transpose(transform_first_marg)
    
    sample_second_marg = transform_first_marg + indep_sample
    return sample_second_marg

#Two functions to initialize the Markov chain with samples evenly spaced
#on a circle.

def two_dim_rota(theta, x=np.asarray([1,0])):
    x_cord = x[0]*np.cos(theta) - x[1]*np.sin(theta)
    y_cord = x[0]*np.sin(theta) + x[1]*np.cos(theta)
    output = np.asarray([x_cord,y_cord])
    return output

def regular_circ_grid(N_grid):
    theta_grid = (2*np.pi/N_grid)*np.arange(N_grid)
    output_store = np.zeros((N_grid, 2))
    for i in range(N_grid):
        output_store[i] = two_dim_rota(theta_grid[i])
    return output_store

def sample_markov_bicoupling(t_grid, couplecov_grid, size_sample=10, 
                             init = 'rand'):
    """
    Sample a Gaussian Markov process over t_grid. 
    The process is characterized by its bivariate couplings. 
    If t_grid is of length n, couplcov_grid is of length n-1.
    At a fixed time, the marginal is of dimension 2.
    ovariance couplings are of dimension 2 by 2.
    For visualization purpose, the first samples can be chosen in a 
    deterministic way. To do so, set init = 'deter'.
    """
    n_times = len(t_grid)
    storage_path = np.zeros((size_sample, n_times, 2))
    if init == 'rand':
        A_0 = couplecov_grid[0][:2, :2]
        current_sample = multivariate_normal(mean=np.zeros(2), cov=A_0, 
                                             size=size_sample)
    else:
        current_sample = 3*regular_circ_grid(N_grid=size_sample)
        
    storage_path[:,0,:] = current_sample
    for it, couple_cov in enumerate(couplecov_grid):
        current_sample = sample_given_marginal(sample_X=current_sample,
                                               couple_cov=couple_cov)
        storage_path[:,it+1,:] = current_sample
    return storage_path

n_margins = 20
t_grid = np.linspace(0, 1, 8)
cplcov_grid_heat = grid_ref_cov_heat(t_grid=t_grid)
sample_markov_bicoupling(t_grid=t_grid, couplecov_grid=cplcov_grid_heat)
