#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:41:54 2026

@author: pfreulon
"""
import numpy as np
import matplotlib.pyplot as plt
import session_info 
import sys
sys.path.append('../Functions/')
from Functions_Gauss_Process_Sampling import *
from Functions_EOT_Ref_Coupling import *
import random
random.seed(1)


#In this script the marginal vectors are of dimension 2

# True diffusion parameters

K = np.array([[3,0],[2,3]]) # Drift matrix
L = np.eye(2)               # Diffusion matrix
A_0 = 10*np.eye(2)          # Covariance at time zero

# Time step parameters

time_init = 0.01
time_final = 1

# Number of samples

n_small, n_mediu, n_large = 100, 500, 1000

n_marg_grid = [n_small, n_mediu, n_large]
time_grid_small = np.linspace(time_init, time_final, num=n_small)
time_grid_mediu = np.linspace(time_init, time_final, num=n_mediu)
time_grid_large = np.linspace(time_init, time_final, num=n_large)

# Computing the true couplings for three different time discretization

couple_grid_OU_small = grid_cplcov_OU(time_grid_small, 
                                      K=K, L=L, A_0=A_0)
couple_grid_OU_mediu = grid_cplcov_OU(time_grid_mediu, 
                                      K=K, L=L, A_0=A_0)
couple_grid_OU_large = grid_cplcov_OU(time_grid_large, 
                                      K=K, L=L, A_0=A_0)
margin_grid_OU_small = []
margin_grid_OU_mediu = []
margin_grid_OU_large = []


for it in range(n_small-1):
    couple_cov = couple_grid_OU_small[it]
    margin_grid_OU_small.append(couple_cov[:2, :2])
    margin_grid_OU_small.append(couple_cov[2:, 2:])
    

for it in range(n_mediu-1):
    couple_cov = couple_grid_OU_mediu[it]
    margin_grid_OU_mediu.append(couple_cov[:2, :2])
    margin_grid_OU_mediu.append(couple_cov[2:, 2:])
    

for it in range(n_large-1):
    couple_cov = couple_grid_OU_large[it]
    margin_grid_OU_large.append(couple_cov[:2, :2])
    margin_grid_OU_large.append(couple_cov[2:, 2:])

#%%

# Low time resolution Error

n_H = 50
H_grid = np.linspace(0.1, 0.9, num=n_H)
eps_grid = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
n_eps = len(eps_grid)

error_cross_cov_small = np.zeros((n_eps, n_H))


for i, eps in enumerate(eps_grid):
    for j, H in enumerate(H_grid):
        grid_ref_cov = grid_ref_cov_fBm(time_grid_small, H=H)
        error = 0
        for k in range(n_small-1):
            Sigma_ref = grid_ref_cov[k]
            couple_cov = couple_grid_OU_small[k]
            A_0 = couple_cov[:2, :2]
            A_1 = couple_cov[2:, 2:]
            Sigma_ref = grid_ref_cov[k]
            eot_couple = cov_eot_refcouple(A = A_0, B = A_1, 
                                         reg=eps, Sigma=Sigma_ref)
            diff_cov = 0.5*(eot_couple - couple_cov)
            error = error + np.linalg.norm(diff_cov)
        error_cross_cov_small[i,j] = error
        


error_cross_cov_mediu = np.zeros((n_eps, n_H))


for i, eps in enumerate(eps_grid):
    for j, H in enumerate(H_grid):
        grid_ref_cov = grid_ref_cov_fBm(time_grid_mediu, H=H)
        error = 0
        for k in range(n_mediu-1):
            Sigma_ref = grid_ref_cov[k]
            couple_cov = couple_grid_OU_mediu[k]
            A_0 = couple_cov[:2, :2]
            A_1 = couple_cov[2:, 2:]
            Sigma_ref = grid_ref_cov[k]
            eot_couple = cov_eot_refcouple(A = A_0, B = A_1, 
                                         reg=eps, Sigma=Sigma_ref)
            diff_cov = 0.5*(eot_couple - couple_cov)
            error = error + np.linalg.norm(diff_cov)
        error_cross_cov_mediu[i,j] = error
        
        

error_cross_cov_large = np.zeros((n_eps, n_H))


for i, eps in enumerate(eps_grid):
    for j, H in enumerate(H_grid):
        grid_ref_cov = grid_ref_cov_fBm(time_grid_large, H=H)
        error = 0
        for k in range(n_large-1):
            Sigma_ref = grid_ref_cov[k]
            couple_cov = couple_grid_OU_large[k]
            A_0 = couple_cov[:2, :2]
            A_1 = couple_cov[2:, 2:]
            Sigma_ref = grid_ref_cov[k]
            eot_couple = cov_eot_refcouple(A = A_0, B = A_1, 
                                         reg=eps, Sigma=Sigma_ref)
            diff_cov = 0.5*(eot_couple - couple_cov)
            error = error + np.linalg.norm(diff_cov)
        error_cross_cov_large[i,j] = error
            
# %%

plt.figure(figsize=(18,5))

# Define a colormap from blue to red
colormap = plt.cm.get_cmap('hot', 9)  # 4 colors from blue to red
plt.subplot(1,3,1)

for i, eps in enumerate(eps_grid):
    error = error_cross_cov_small[i, :]

    # Set color: black for the first plot, colormap for the rest
    if i == 0:
        color = 'black'
    else:
        color = colormap(i)  # i-1 because the first plot is black

    plt.plot(H_grid, error, color=color, label=r'$\varepsilon=$' + str(eps))

plt.legend(loc='best', fontsize=15)
plt.ylabel('Error',  size=15)
plt.xlabel('H', size=15)
plt.ylim(1.5, 3)
plt.xticks(size=12)
plt.yticks(size=12)



# Define a colormap from blue to red
colormap = plt.cm.get_cmap('hot', 9)  # 4 colors from blue to red
plt.subplot(1,3,2)
for i, eps in enumerate(eps_grid):
    error = error_cross_cov_mediu[i, :]

    # Set color: black for the first plot, colormap for the rest
    if i == 0:
        color = 'black'
    else:
        color = colormap(i)  # i-1 because the first plot is black

    plt.plot(H_grid, error, color=color, label=r'$\varepsilon=$' + str(eps))

plt.xlabel('H', size=15)
plt.xticks(size=12)
plt.yticks(size=12)
plt.ylim(1.5, 6)

plt.subplot(1,3,3)
# Define a colormap from blue to red
colormap = plt.cm.get_cmap('hot', 9)  # 4 colors from blue to red

for i, eps in enumerate(eps_grid):
    error = error_cross_cov_large[i, :]

    # Set color: black for the first plot, colormap for the rest
    if i == 0:
        color = 'black'
    else:
        color = colormap(i)  # i-1 because the first plot is black

    plt.plot(H_grid, error, color=color, label=r'$\varepsilon=$' + str(eps))

plt.xlabel('H', size=15)
plt.ylim(1.5, 9)
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig('../Figures_EOTReference/Error_cov_fBm.pdf')
        
    