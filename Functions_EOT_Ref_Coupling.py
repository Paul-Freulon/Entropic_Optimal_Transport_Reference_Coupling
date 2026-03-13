#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:23:34 2026

@author: freulon
"""
import numpy as np
from scipy.linalg import pinv
from scipy.linalg import sqrtm


def cov_eot_refcouple(A, B, reg=0, Sigma='prod'):
    """
    Return the covariance matrix of the transport plan between two Gaussians
    with zero mean and respective covariance matrices A and B.
    
    In the case reg=0, the function returns the covariance solution of the 
    classic optimal transport between Gaussians.
    
    If reg is non zero and reference covariance coupling is not specified,
    we return the solution to classic entropic OT. That is with product of 
    input as reference coupling.
    
    Sigma is the covariance matrix of the refence coupling.
    """
    dim = A.shape[0]
    sqrt_A = sqrtm(A)
    if type(Sigma) == str:
        Sigma = np.block([[A, np.zeros((dim,dim))],
                         [ np.zeros((dim,dim)),B]])
    Gamma = pinv(Sigma)
    M = np.eye(N=dim) - reg*Gamma[:dim, dim:]
    M_t = np.transpose(M)
    
    #Computing A^{1/2}MBM^TA^{1/2}
    
    prod_mat = sqrt_A @ M @ B @ M_t @ sqrt_A
    
    geo_avg_reg = sqrtm(prod_mat + (reg/2)**2 * np.eye(N=dim))
    
    C_eps = sqrt_A @ geo_avg_reg @ pinv(sqrt_A)    
    
    C_eps = (C_eps - (reg/2)*np.eye(N=dim)) @ pinv(M_t)

    cov_mat = np.block([[A, C_eps],
                        [np.transpose(C_eps),B]])
    return cov_mat

#test function
d = 2
eps = 1
A = np.eye(d)
B = 2*np.eye(d)

#computation of a reference covariance
Sigma=np.zeros((2*d,2*d))
for i in range(2*d):
    for j in range(2*d):
        Sigma[i,j] = np.min((i+1, j+1))/np.max((i+1, j+1))
print('Reference covariance coupling')
print(Sigma)
print('Entropic OT with or without reference coupling')
eot_couple = cov_eot_refcouple(A=A, B=B, reg=eps, Sigma=Sigma)
print(eot_couple)


def cov_eot_onedim(var_a=1, var_b=1, eps=0, rho=0):
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

def cost_eot_onedim(var_a=1, var_b=1, eps=0, rho=0): 
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

