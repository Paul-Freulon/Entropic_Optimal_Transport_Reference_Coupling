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


def cov_eot_prior(A, B, reg=0, Sigma=np.eye(4)):
    """
    Return the covariance matrix of the transport plan between two Gaussians
    with zero mean and respective covariance matrices A and B.
    
    In the case reg=0, the function returns the covariance solution of the 
    classic optimal transport between Gaussians.
    
    If reg is non zero and reference covariance matrix is diag(A,B),
    we recover the solution to classic entropic OT.
    
    Sigma is the covariance matrix of the refence coupling.
    """
    dim_marge = A.shape[0]
    sqrt_A = sqrtm(A)
    Gamma = pinv(Sigma)
    M = np.eye(N=dim_marge) - reg*Gamma[:dim_marge, dim_marge:]
    M_t = np.transpose(M)
    
    #Computing A^{1/2}MBM^TA^{1/2}
    
    prod_mat = sqrt_A @ M @ B @ M_t @ sqrt_A
    
    geo_avg_reg = sqrtm(prod_mat + (reg/2)**2 * np.eye(N=dim_marge))
    
    C_eps = sqrt_A @ geo_avg_reg @ pinv(sqrt_A)    
    
    C_eps = (C_eps - (reg/2)*np.eye(N=dim_marge)) @ pinv(M_t)

    cov_mat = np.block([[A, C_eps],
                        [np.transpose(C_eps),B]])
    return cov_mat

def integrate_expmat_OU(time, drift_mat, diff_mat):
    """
    If  K = drift_mat and L = diff_mat,
    for the purpose of studying the SDE: dX_t = -K X_t dt + L dW_t,
    we compute the integral int_0^time exp(s K)LL^T exp(sK^T)ds.
    """
    time_grid = np.linspace(0, time, num=int(np.ceil(time*100)))
    delta_t = time_grid[1]-time_grid[0]
    grid_exp_mat = [expm(s*drift_mat)@diff_mat for s in time_grid]
    grid_to_sum = [M@np.transpose(M) for M in grid_exp_mat]
    output = delta_t*np.sum(grid_to_sum,axis=0)
    return output



def var_mat_OU(time, drift_mat, diff_mat, init_mat=np.array([[0,0],[0,0]])):
    """
    Set K = drift_mat and L=diff_mat.
    Study the SDE dX_t = -K X_t dt + L dW_t 
    With inital condition X_0 sim N(A_0) where A_0 = init_mat
    the output is the 2 times 2 covariance matrix of X_t.
    """
    int_expmat = integrate_expmat_OU(time=time, drift_mat=drift_mat,
                                     diff_mat=diff_mat)
    exp_drift = expm(-time * drift_mat)
    output = exp_drift @ (init_mat + int_expmat) @ np.transpose(exp_drift)
    #additional step to insure symmetry 
    output = (1/2)*(output + np.transpose(output))

    return output


def cross_cov_OU(t_0, t_1, drift_mat, diff_mat,
                 init_mat=np.array([[0,0],[0,0]])):
    """
    K = drift_mat, L = diff_mat.
    The SDE is dX_t = -F X_t dt + L dW_t
    The output is the cross covariance between X_{t_0} and X_{t_1}.
    That is E(X_{t_0}X_{t_1}^T).
    """
    var_mat = var_mat_OU(time=t_0, drift_mat=drift_mat, 
                         diff_mat=diff_mat, init_mat=init_mat)
    right_mat = expm(-(t_1 - t_0)*np.transpose(drift_mat))

    cross_cov = var_mat @ right_mat
    return cross_cov



def cov_two_times_OU(t_0, t_1, drift_mat, diff_mat, 
                     init_mat=np.array([[0,0],[0,0]])):
    """
    Compute the covariance matrix of the vector (X_{t_0}, X_{t_1}) \in R^4.
    """
    var_0 = var_mat_OU(time=t_0, drift_mat=drift_mat, diff_mat=diff_mat,
                       init_mat=init_mat)
    var_1 = var_mat_OU(time=t_1,  drift_mat=drift_mat, diff_mat=diff_mat,
                       init_mat=init_mat)
    cov_01 = cross_cov_OU(t_0=t_0, t_1=t_1, drift_mat=drift_mat,
                          diff_mat=diff_mat, init_mat=init_mat)
    full_cov = np.block([[var_0, cov_01],
                        [np.transpose(cov_01),var_1]])
    

    return full_cov


def grid_marge_var(t_grid, drift_mat, diff_mat, init_mat=np.eye(2)):
    """
    Compute the marginal covariances at the n times of t_grid.
    """
    marginal_storage = []
    for time in t_grid:
        var_marg = var_mat_OU(time=time, drift_mat=drift_mat, 
                   diff_mat=diff_mat, 
                   init_mat=init_mat)
        marginal_storage.append(var_marg)
    
    return marginal_storage

def grid_ref_cov_OU(t_grid, drift_mat, diff_mat,
                 init_mat=np.array([[0,0],[0,0]])):
    """
    If there are n different times in t_grid, this function computes the 
    n-1 covariances coupling successive times for the OU dynamic. 
    It returns the list of the n-1 reference coupling covariances.
    """
    ref_cov_storage = []
    n = len(t_grid)
    for i in range(n-1):
        t_0 = t_grid[i]
        t_1 = t_grid[i+1]
        ref_cov = cov_two_times_OU(t_0=t_0, t_1=t_1, drift_mat=drift_mat,
                                   diff_mat = diff_mat, init_mat=init_mat)
        ref_cov_storage.append(ref_cov)
    return ref_cov_storage

def kernel_fBm(t_min, t_max, H=1/2):
    """
    Scalar kernel for the fractional Brownian motion.
    """
    temp = (t_max+1)**(2*H) + (t_min+1)**(2*H) - (t_max-t_min)**(2*H)
    const = 2 * (t_max+1)**(2*H)
    return temp/const

def couple_cov_fBm(t_min, t_max, H=1/2):
    """
    Coupling covariance for 2 dimensional fractional Brownian motion
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
    for i in range(n-1):
        t_0 = t_grid[i]
        t_1 = t_grid[i+1]
        ref_cov = couple_cov_fBm(t_min=t_0, t_max=t_1, H=H)
        ref_cov_storage.append(ref_cov)
    return ref_cov_storage

def kernel_heat(t_min, t_max, sigma=1):
    """
    Scalar heat kernel rho(t,s)=exp(-(t-s)^2/(2sigma^2)).
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

def sample_given_marginal(sample_first_marg, couple_cov):
    """
    Allows to sample from couple_cov given samples from the first marginal.
    We suppose that sample_first_marg has covariance the top left block of 
    couple_cov.
    """
    n_sample = sample_first_marg.shape[0]
    
    A = couple_cov[:2, :2]
    B = couple_cov[2:, 2:]
    C = couple_cov[:2, 2:]
    schur_comp = B - np.transpose(C) @ pinv(A) @ C

    indep_sample = multivariate_normal(mean=np.zeros(2), cov=schur_comp,
                                       size=n_sample)
    transform_first_marg = pinv(A) @ np.transpose(sample_first_marg)
    transform_first_marg = np.transpose(C) @ transform_first_marg 
    transform_first_marg = np.transpose(transform_first_marg)
    
    sample_second_marg = transform_first_marg + indep_sample
    return sample_second_marg

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

def sample_markov_bicoupling(t_grid, couplcov_grid, size_sample=10):
    """
    Sample a Gaussian Markov process over t_grid. 
    The process is characterized by its bivariate couplings. 
    If t_grid is of length n, couplcov_grid is of length n-1.
    At a fixed time, the marginal is of dimension 2. Thus, a covariance 
    coupling is of dimension 2 by 2.
    """
    n_times = len(t_grid)
    storage_path = np.zeros((size_sample, n_times, 2))
    current_sample = 3*regular_circ_grid(N_grid=size_sample)
    storage_path[:,0,:] = current_sample
    for it, couple_cov in enumerate(couplcov_grid):
        current_sample = sample_given_marginal(sample_first_marg=current_sample,
                                               couple_cov=couple_cov)
        storage_path[:,it+1,:] = current_sample
    return storage_path

