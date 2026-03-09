#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:25:13 2026

@author: freulon
"""

#Modules

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from Functions_Gauss_Process_PEOT import *
import random

#In this script the marginal vectors are of dimension 2.


# Set parameter

random.seed(42)

# True diffusion parameters

Drift = np.array([[2.5,0],[1,2.5]]) # Drift matrix
Diffu = 0.5*np.eye(2)               # Diffusion matrix
init_var= 10*np.eye(2)          # Covariance at time zero

# Time step parameters

n_times = 100
time_grid = np.linspace(0.01, 1, num=n_times) 


#%% 

#Time marginal
marginal_grid = grid_marge_var(t_grid = time_grid, drift_mat= Drift, 
                               diff_mat=Diffu, init_mat= init_var)

#Reference coupling with fractional Brownian motion kernel
H = 0.95 #Hurst index of the fBm

ref_cov_grid_fBm = grid_ref_cov_fBm(t_grid=time_grid, H=H)

#Reference coupling with heat kernel

ref_cov_grid_heat = grid_ref_cov_heat(t_grid=time_grid)




