#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 17:21:21 2026

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
sys.path.append('../Functions')
from Functions_Gauss_Process_Sampling import *
from Functions_EOT_Ref_Coupling import *
import random
random.seed(1)

#session_info.show(html=False)

t_start = time()

#In this script the marginal vectors are of dimension 2

# True diffusion parameters

K = np.array([[3,0],[2,3]]) # Drift matrix
L = np.eye(2)               # Diffusion matrix
A_0 = 10*np.eye(2)          # Covariance at time zero

# Time step parameters

n_times = 100
time_init = 0.01
time_final = 1
time_grid = np.linspace(time_init, time_final, num=n_times) 


#Compute the marginal covariances given by the Ornstein-Uhlenbeck

marginal_cov_grid = grid_margevar_OU(t_grid = time_grid, K=K, L=L, A_0=A_0)
# Number of samples
n_sample = 10


# Sampling from the true diffusion

print('Sampling from the true diffusion')

n_marg_grid = [100, 500, 1000]
sample_truepath_time = []
for it, n_marg in enumerate(n_marg_grid):
#times where the marginals are available
    loc_time_grid = np.linspace(time_init, time_final, num=n_marg)
    
    couple_grid_OU = grid_cplcov_OU(loc_time_grid, K=K, L=L, A_0=A_0)
    
    sample = sample_markov_bicoupling(loc_time_grid, couple_grid_OU, 
                                      size_sample=n_sample, init='deter') 
    sample_truepath_time.append(sample)
# Display paths of the diffusion

#3D display

path_storage = sample_truepath_time[0]

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
for i in range(n_sample):
    ax.plot(time_grid, path_storage[i,:,0], path_storage[i,:,1])
ax.set_xlabel("time")
ax.set_ylabel("x")
ax.set_zlabel("y")
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
plt.savefig('../Figures_EOTReference/sample_diffpaths_3d.pdf', format='pdf')

#2D display

fig = plt.figure(figsize=(18,11))
for it, n_marg in enumerate(n_marg_grid):
    path = sample_truepath_time[it]
    plt.subplot(2,3,it+1)
    plt.title("Number of margins = {}".format(n_marg), size=14)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel('x', size=15)
    plt.ylabel('y', size=15)
    for i in range(n_sample):
        plt.plot(path[i,:,0], path[i,:,1])
    plt.subplot(2,3,it+4)
    for i in range(n_sample):
        plt.plot(np.linspace(time_init, time_final, num=n_marg), path[i,:,0])
        plt.ylim(-3, 3)
        plt.xlabel('time', size=15)
        plt.ylabel('x', size=15)
    plt.savefig('../Figures_EOTReference/2DPath_diffusion_timediscr.pdf',
                format='pdf')



# Impact of time discretization for independent, Brownian motion,
# fractional Brownian motion, and heat kernel chosen as reference couplings

init_cov = 10*np.eye(2)          # Covariance at time zero
eps = 0.1
H = 0.25
n_marg_grid = [100,500,1000]
sample_indep_time = []
sample_brown_time = []
sample_fBmH_time = []
sample_heat_time = []

for it, n_marg in enumerate(n_marg_grid):
    #times where the marginals are available
    loc_time_grid = np.linspace(time_init, time_final, num=n_marg)
    
    #Marginal covariances
    marginal_grid = grid_margevar_OU(t_grid=loc_time_grid, K=K, 
                   L=L, A_0= init_cov)
    
    eot_prod_storage = []
    eot_brown_storage = []
    eot_fBmH_storage = []
    eot_heat_storage = []
    
    brown_refe_grid = grid_ref_cov_fBm(loc_time_grid)
    fBmH_refe_grid = grid_ref_cov_fBm(loc_time_grid, H=H)
    heat_refe_grid = grid_ref_cov_heat(loc_time_grid)
    
    for it in range(n_marg-1):
        A_0 = marginal_grid[it]
        A_1 = marginal_grid[it+1]

        #Entropic OT with independent couplings
        
        eot_prod = cov_eot_refcouple(A = A_0, B = A_1, reg=eps) 
        eot_prod_storage.append(eot_prod)
    
        #Entropic OT coupling with Brownian motion reference
    
        brown_couple_cov = brown_refe_grid[it]
        eot_refe_brown = cov_eot_refcouple(A = A_0, B = A_1, reg=eps, 
                                  Sigma = brown_couple_cov)
        eot_brown_storage.append(eot_refe_brown)
        
        #Entropic OT coupling with fractional brownian motion reference
        
        fbmH_couple_cov = fBmH_refe_grid[it]
        eot_refe_fbmH = cov_eot_refcouple(A = A_0, B = A_1, reg=eps, 
                                  Sigma = fbmH_couple_cov)
        eot_fBmH_storage.append(eot_refe_fbmH)
        
        #Entropic OT coupling with heat kernel reference
        heat_couple_cov = heat_refe_grid[it]
        eot_refe_heat = cov_eot_refcouple(A = A_0, B = A_1, reg=eps, 
                                  Sigma = heat_couple_cov)
        eot_heat_storage.append(eot_refe_heat)

    #Sampling and storing
    
    path_eot_prod = sample_markov_bicoupling(loc_time_grid,
                                              eot_prod_storage,
                                              size_sample=n_sample,
                                              init='deter')
    sample_indep_time.append(path_eot_prod)
    
    path_eot_brown = sample_markov_bicoupling(loc_time_grid,
                                              eot_brown_storage,
                                              size_sample=n_sample,
                                              init='deter')
    sample_brown_time.append(path_eot_brown)
    
    path_eot_fbmH = sample_markov_bicoupling(loc_time_grid,
                                              eot_fBmH_storage,
                                              size_sample=n_sample,
                                              init='deter')
    sample_fBmH_time.append(path_eot_fbmH)
    
    path_eot_heat = sample_markov_bicoupling(loc_time_grid,
                                              eot_heat_storage,
                                              size_sample=n_sample,
                                              init='deter')
    sample_heat_time.append(path_eot_heat)
    
# Display paths

fig = plt.figure(figsize=(18,11))

for it, n_marg in enumerate(n_marg_grid):
    path = sample_indep_time[it]
    plt.subplot(2,3,it+1)
    plt.title("Number of margins = {}".format(n_marg), size=14)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel('x', size=15)
    plt.ylabel('y', size=15)
    for i in range(n_sample):
        plt.plot(path[i,:,0], path[i,:,1])
    plt.subplot(2,3,it+4)
    plt.ylim(-4, 4)
    plt.xlabel('time', size=15)
    plt.ylabel('x', size=15)
    for i in range(n_sample):
        plt.plot(np.linspace(time_init, time_final, num=n_marg), path[i,:,0])
    plt.savefig('../Figures_EOTReference/2DPath_indep_ref_timediscr.pdf', 
                format='pdf')


fig = plt.figure(figsize=(18,11))
for it, n_marg in enumerate(n_marg_grid):
    path = sample_brown_time[it]
    plt.subplot(2,3,it+1)
    plt.title("Number of margins = {}".format(n_marg), size=14)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel('x', size=15)
    plt.ylabel('y', size=15)
    for i in range(n_sample):
        plt.plot(path[i,:,0], path[i,:,1])
    plt.subplot(2,3,it+4)
    plt.ylim(-4, 4)
    plt.xlabel('time', size=15)
    plt.ylabel('x', size=15)
    for i in range(n_sample):
         plt.plot(np.linspace(time_init, time_final, num=n_marg), path[i,:,0])
    plt.savefig('../Figures_EOTReference/2DPath_brown_ref_timediscr.pdf', 
                format='pdf')
    
fig = plt.figure(figsize=(18,11))
for it, n_marg in enumerate(n_marg_grid):
    path = sample_fBmH_time[it]
    plt.subplot(2,3,it+1)
    plt.title("Number of margins = {}".format(n_marg), size=14)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel('x', size=15)
    plt.ylabel('y', size=15)
    for i in range(n_sample):
        plt.plot(path[i,:,0], path[i,:,1])
    plt.subplot(2,3,it+4)
    plt.ylim(-4, 4)
    plt.xlabel('time', size=15)
    plt.ylabel('x', size=15)
    for i in range(n_sample):
         plt.plot(np.linspace(time_init, time_final, num=n_marg), path[i,:,0])
    plt.savefig('../Figures_EOTReference/2DPath_fBmH_ref_timediscr.pdf', 
                format='pdf')


fig = plt.figure(figsize=(18,11))
for it, n_marg in enumerate(n_marg_grid):
    path = sample_heat_time[it]
    plt.subplot(2,3,it+1)
    plt.title("Number of margins = {}".format(n_marg), size=14)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel('x', size=15)
    plt.ylabel('y', size=15)
    for i in range(n_sample):
        plt.plot(path[i,:,0], path[i,:,1])
    plt.subplot(2,3,it+4)
    plt.xlabel('time', size=15)
    plt.ylabel('x', size=15)
    for i in range(n_sample):
        plt.ylim(-4, 4)
        plt.plot(np.linspace(time_init, time_final, num=n_marg), path[i,:,0])
    plt.savefig('../Figures_EOTReference/2DPath_heat_ref_timediscr.pdf', 
                format='pdf')

#  3D display of paths built from entropic OT couplings 
k = 0 #scenario for the number of marginals: 0 -> 100, 1 -> 500, 2 -> 1000
time_grid = np.linspace(time_init, time_final, num=n_marg_grid[k])

#Independent reference
paths = sample_indep_time[k]
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

for i in range(n_sample):
   ax.plot(time_grid, paths[i,:,0], paths[i,:,1])
ax.set_xlabel("time")
ax.set_ylabel("x")
ax.set_zlabel("y")
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])

ax.set_title("Independent reference "+
             r"and $\varepsilon$ = {}".format(eps))
plt.savefig('../Figures_EOTReference/Path_3D_EOT_Indepref.pdf', 
            format='pdf')

# Brownian motion reference
paths = sample_brown_time[k]
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

for i in range(n_sample):
    ax.plot(time_grid, paths[i,:,0],
            paths[i,:,1])

ax.set_xlabel("time")
ax.set_ylabel("x")
ax.set_zlabel("y")

ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
ax.set_title("Brownian motion "+
             r"reference and $\varepsilon$ = {}".format(eps))
plt.savefig('../Figures_EOTReference/Paths_3D_EOT_brown_ref.pdf',
            format='pdf')

# fraction Brownian motion reference
paths = sample_fBmH_time[k]
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

for i in range(n_sample):
    ax.plot(time_grid, paths[i,:,0],
            paths[i,:,1])

ax.set_xlabel("time")
ax.set_ylabel("x")
ax.set_zlabel("y")

ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
ax.set_title("Fractional Brownian motion - H={}".format(H)+
             r" and $\varepsilon$ = {}".format(eps))
plt.savefig('../Figures_EOTReference/Paths_3D_fBmH_ref.pdf',
            format='pdf')

# fraction Brownian motion reference
paths = sample_heat_time[k]
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

for i in range(n_sample):
    ax.plot(time_grid, paths[i,:,0],
            paths[i,:,1])

ax.set_xlabel("time")
ax.set_ylabel("x")
ax.set_zlabel("y")

ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
ax.set_title("Heat kernel reference"
             r" and $\varepsilon$ = {}".format(eps))
plt.savefig('../Figures_EOTReference/Paths_3D_heat_ref.pdf',
            format='pdf')

print('Time to run the script :')
print(time()-t_start)
