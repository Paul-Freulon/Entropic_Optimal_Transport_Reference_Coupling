#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:44:16 2026

@author: freulon
"""


import numpy as np
import matplotlib.pyplot as plt
import session_info 
from numpy.random import multivariate_normal
from scipy.linalg import sqrtm
from scipy.linalg import expm
from scipy.linalg import pinv
from numpy.linalg import eigh
from time import time
import sys
sys.path.append('../Functions/')
from Functions_Gauss_Process_Sampling import *
from Functions_EOT_Ref_Coupling import *
import random
random.seed(1)

session_info.show(html=False)

#In this script the marginal vectors are of dimension 2

# True diffusion parameters

K = np.array([[3,0],[2,3]]) # Drift matrix
L = np.eye(2)               # Diffusion matrix
init_var = 10*np.eye(2)          # Covariance at time zero

# Time step parameters

time_init = 0.01
time_final = 1

# Number of samples
n_sample = 10

#%Mixed smoothness kernel reference

eps=0.1
n_marg_grid = [100,500,1000]
sample_mixsmooth_time = []

for it, n_marg in enumerate(n_marg_grid):
    #times where the marginals are available
    loc_time_grid = np.linspace(time_init, time_final, num=n_marg)
    
    #Marginal covariances
    
    t_start = time()
    marginal_grid = grid_margevar_OU(t_grid=loc_time_grid, K=K, 
                   L=L, A_0= init_var)
    print('Time to compute the {} marginals:'.format(n_marg))
    print(time()-t_start)
    
    eot_mix_storage = []
    mix_refe_grid = grid_ref_covmix(loc_time_grid)
    
    t_start = time()
    for it in range(n_marg-1):
        A_0 = marginal_grid[it]
        A_1 = marginal_grid[it+1]
    
        #Entropic OT coupling with mix kernel reference
    
        mix_couple_cov = mix_refe_grid[it]
        eot_refe_mix = cov_eot_refcouple(A = A_0, B = A_1, reg=eps, 
                                  Sigma = mix_couple_cov)
        eot_mix_storage.append(eot_refe_mix)
    print('Time to compute the {} reference couplings'.format(n_marg-1)+
          ' and to solve the {} entropic OT problems:'.format(n_marg-1))  
    print(time()-t_start)


    #Sampling and storing
    
    t_start = time()
    path_eot_mix = sample_markov_bicoupling(loc_time_grid,
                                              eot_mix_storage,
                                              size_sample=n_sample,
                                              init='deter')
    print('Time to sample {} trajectories'.format(n_sample) +
          ' with {} time marginals:'.format(n_marg))
    print(time()-t_start)

    sample_mixsmooth_time.append(path_eot_mix)
    
    
#%% Display paths

fig = plt.figure(figsize=(18,18))

for it, n_marg in enumerate(n_marg_grid):
    path = sample_mixsmooth_time[it]
    plt.subplot(3,3,it+1)
    plt.title("Number of margins = {}".format(n_marg), size=14)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel('x', size=15)
    plt.ylabel('y', size=15)
    for i in range(n_sample):
        plt.plot(path[i,:,0], path[i,:,1])
    plt.subplot(3,3,it+4)
    plt.ylim(-4, 4)
    plt.xlabel('time', size=15)
    plt.ylabel('x', size=15)
    for i in range(n_sample):
        plt.plot(np.linspace(time_init, time_final, num=n_marg), path[i,:,0])
    plt.subplot(3,3,it+7)
    plt.ylim(-4, 4)
    plt.xlabel('time', size=15)
    plt.ylabel('y', size=15)
    for i in range(n_sample):
        plt.plot(np.linspace(time_init, time_final, num=n_marg), path[i,:,1])
    plt.savefig('../Figures_EOTReference/2DPath_mix_ref_timediscr.pdf', 
                format='pdf')
    
