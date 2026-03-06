#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 17:21:21 2026

@author: freulon
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from scipy.linalg import sqrtm
from scipy.linalg import expm
from scipy.linalg import pinv
from numpy.linalg import eigh
from Functions_Gauss_Process_PEOT import *
import random


#In this script the marginal vectors are of dimension 2.


# Set parameter

random.seed(42)

# True diffusion parameters

Drift = np.array([[2.5,0],[1,2.5]]) # Drift matrix
Diffu = 0.5*np.eye(2)               # Diffusion matrix
init_var= 10*np.eye(2)          # Covariance at time zero

# Prior diffusion parameters

Drift_prior =  np.array([[2,0],[0,2]]) # Prior drift matrix
Diffu_prior = 2*np.eye(2)              # Prior diffusion matrix
init_var_prior = 5*np.eye(2)       # Prior initial covariance matrix

# Time step parameters

time = 0.5
time_bis = 2
n_times = 100
time_grid = np.linspace(0.01, 1, num=n_times) 

#%%

#One time marginal test

var_OU_onetime = var_mat_OU(time=time, drift_mat=Drift, diff_mat=Diffu, 
                init_mat=init_var)

print('Variance at one time')
print(var_OU_onetime)

#Two times marginal test

print('Full covariance matrix of the two dimensional Ornstein-Uhlenbeck\
 Process between time t_0 = {} and time t_1 = {}.'.format(time, time_bis))
cov_OU_2times = cov_two_times_OU(t_0=time, t_1=time_bis, drift_mat=Drift, 
                       diff_mat=Diffu,init_mat=init_var)
print(cov_OU_2times)
print('test symmetry coupling covariance')
print(cov_OU_2times == np.transpose(cov_OU_2times))
print('test positivity: eigen values')
print(eigh(cov_OU_2times)[0])

# %%

#Test sampling
n_sample = 10

#Compute the coupling covariances between times (t_i,t_i+1) for all i.
prior_cov_grid = grid_ref_cov_OU(time_grid, drift_mat=Drift, diff_mat=Diffu, 
                              init_mat=init_var)

misspec_prior_grid = grid_ref_cov_OU(time_grid, drift_mat=Drift_prior, 
                                 diff_mat=Diffu_prior, 
                                init_mat=init_var_prior)

couple_cov = prior_cov_grid[0]
cov_first_marge = couple_cov[:2,:2]
sample_first_marg = multivariate_normal(mean=np.zeros(2),
                                  cov = cov_first_marge,
                                  size=n_sample)

print('test sample given marginal')
print(sample_given_marginal(sample_first_marg=sample_first_marg, 
                                couple_cov=couple_cov))


print('test sample markov bicoupling')
path_storage = sample_markov_bicoupling(time_grid, prior_cov_grid, 
                                        size_sample=n_sample)


print('Sample from the true diffusion with varying time grid')

n_marg_grid = [100,500,1000]
sample_truepath_time = []
for it, n_marg in enumerate(n_marg_grid):
#times where the marginals are available
    loc_time_grid = np.linspace(0.1, 1, num=n_marg)
    
    prior_grid = grid_ref_cov_OU(loc_time_grid, 
                                      drift_mat=Drift, 
                                      diff_mat=Diffu, 
                                      init_mat=init_var)
    
    sample = sample_markov_bicoupling(loc_time_grid, prior_grid, 
                                            size_sample=n_sample) 
    sample_truepath_time.append(sample)
    
    
# %%

#Test Gaussian entropic optimal transport

#Computation one coupling

marginal_grid = grid_marge_var(t_grid = time_grid, drift_mat= Drift, 
                               diff_mat=Diffu, init_mat= init_var)
A_0 = marginal_grid[0]
A_1 = marginal_grid[1]

print('Marginal covariance matrices')
print(A_0)
print(A_1)

eps = 2
eps_grid = [0.01, 0.1, 1]
eot_prod = cov_eot_prior(A = A_0, B = A_1, reg=eps) 

print('Entropic OT plan with eps = {} and independent coupling'.format(eps))
print(eot_prod)

eot_prior = cov_eot_prior(A = A_0, B = A_1, reg=eps, Sigma = couple_cov)


print('Entropic OT plan with eps = {} and correct prior coupling'.format(eps))
print(eot_prior)

#Computation of all couplings.

noreg_ot_storage = []
eot_prod_storage = []
eot_prior_storage = []
eot_misspecified_storage = []

for it in range(n_times-1):
    A_0 = marginal_grid[it]
    A_1 = marginal_grid[it+1]
    
    #unregularized optimal transport couplings
    ot_couple = cov_eot_prior(A = A_0, B = A_1, reg=0)
    noreg_ot_storage.append(ot_couple) 
    
    #independent couplings
    
    eot_prod = cov_eot_prior(A = A_0, B = A_1, reg=eps) 
    eot_prod_storage.append(eot_prod)
    
    #prior coupling
    prior_cov = prior_cov_grid[it]
    eot_prior = cov_eot_prior(A = A_0, B = A_1, reg=eps, Sigma = prior_cov)
    eot_prior_storage.append(eot_prior)
    
    #Misspecified prior coupling
    miss_prior = misspec_prior_grid[it] 
    eot_missprior = cov_eot_prior(A = A_0, B = A_1, reg=eps, 
                                  Sigma = miss_prior)
    eot_misspecified_storage.append(eot_missprior)
    
#For entropic optimal transport, computations of couplings with different
#values of epsilon

eot_prod_storage_eps = []
eot_prior_storage_eps = []
eot_missprior_storage_eps = []
    
for vareps in eps_grid:
    
    prod_storage_eps = []
    prior_storage_eps = []
    missprior_storage_eps = []
    
    for it in range(n_times-1):
        A_0 = marginal_grid[it]
        A_1 = marginal_grid[it+1]
        
        #Independent couplings
        eot_prod = cov_eot_prior(A = A_0, B = A_1, reg=vareps) 
        prod_storage_eps.append(eot_prod)
        
        #Prior coupling
        prior_cov = prior_cov_grid[it]
        eot_prior = cov_eot_prior(A = A_0, B = A_1, reg=vareps,
                                  Sigma = prior_cov)
        prior_storage_eps.append(eot_prior)
        
        #Misspecified prior couplings
        missprior_cov = misspec_prior_grid[it]
        eot_missprior = cov_eot_prior(A = A_0, B = A_1, reg=vareps, 
                                      Sigma = missprior_cov)
        missprior_storage_eps.append(eot_missprior)
    
    eot_prod_storage_eps.append(prod_storage_eps)
    eot_prior_storage_eps.append(prior_storage_eps)
    eot_missprior_storage_eps.append(missprior_storage_eps)
    
# %% Sample from optimal transport couplings

#Classic EOT


path_storage_EOTindep = sample_markov_bicoupling(time_grid,
                                                 eot_prod_storage, 
                                        size_sample=n_sample)

#EOT with prior

path_storage_EOT_prior = sample_markov_bicoupling(time_grid, 
                                                  eot_prior_storage,
                                                  size_sample=n_sample)

#EOT with misspecified prior

path_EOT_missprior = sample_markov_bicoupling(time_grid, 
                                              eot_misspecified_storage,
                                              size_sample=n_sample)

#Classic optimal transport without regularization

path_storage_otnoreg = sample_markov_bicoupling(time_grid, noreg_ot_storage,
                                                  size_sample=n_sample)

# %% Sample from entropic OT with different values of epsilon

#EOT with independent coupling

path_EOTindep_Smalleps = sample_markov_bicoupling(time_grid, 
                                                  eot_prod_storage_eps[0], 
                                                  size_sample=n_sample)
path_EOTindep_Medeps = sample_markov_bicoupling(time_grid, 
                                                  eot_prod_storage_eps[1], 
                                                  size_sample=n_sample)
path_EOTindep_Bigeps = sample_markov_bicoupling(time_grid, 
                                                  eot_prod_storage_eps[2], 
                                                  size_sample=n_sample)
path_EOTindep_vareps = [path_EOTindep_Smalleps, path_EOTindep_Medeps,
                        path_EOTindep_Bigeps]

#EOT with prior

path_EOTprior_Smalleps = sample_markov_bicoupling(time_grid,
                                                  eot_prior_storage_eps[0],
                                                  size_sample=n_sample)

path_EOTprior_Medeps = sample_markov_bicoupling(time_grid,
                                                  eot_prior_storage_eps[1],
                                                  size_sample=n_sample)

path_EOTprior_Bigeps = sample_markov_bicoupling(time_grid,
                                                  eot_prior_storage_eps[2],
                                                  size_sample=n_sample)

path_EOTprior_vareps = [path_EOTprior_Smalleps, path_EOTprior_Medeps,
                        path_EOTprior_Bigeps]

#EOT with misspecified prior

path_EOTmissprior_Smalleps = sample_markov_bicoupling(time_grid,
                                                  eot_missprior_storage_eps[0],
                                                  size_sample=n_sample)

path_EOTmissprior_Medeps = sample_markov_bicoupling(time_grid,
                                                  eot_missprior_storage_eps[1],
                                                  size_sample=n_sample)

path_EOTmissprior_Bigeps = sample_markov_bicoupling(time_grid,
                                                  eot_missprior_storage_eps[2],
                                                  size_sample=n_sample)

path_EOTmissprior_vareps = [path_EOTmissprior_Smalleps, 
                            path_EOTmissprior_Medeps,
                            path_EOTmissprior_Bigeps]

# %% Display paths of the diffusion

#3D display
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

for i in range(n_sample):
    ax.plot(time_grid, path_storage[i,:,0], path_storage[i,:,1])

ax.set_xlabel("time")
ax.set_ylabel("x")
ax.set_zlabel("y")


ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])

ax.set_title("Paths constructed from the diffusion")
plt.savefig('Figures_EOTReference/sample_diffpaths_3d.pdf', format='pdf')

plt.figure(figsize=(6,5))
plt.title("Number of margins = {}".format(n_times), size=14)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
for i in range(n_sample):
    plt.plot(path_storage[i,:,0], path_storage[i,:,1])
plt.savefig('Figures_EOTReference/2DPath_diffusion.pdf', format='pdf')



fig = plt.figure(figsize=(18,5))
plt.suptitle(r"Path sampled from the diffusion",
             size=20)
for it, n_marg in enumerate(n_marg_grid):
    path = sample_truepath_time[it]
    plt.subplot(1,3,it+1)
    plt.title("Number of margins = {}".format(n_marg), size=14)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    for i in range(n_sample):
        plt.plot(path[i,:,0], path[i,:,1])
    plt.savefig('Figures_EOTReference/2DPath_diffusion_timediscr.pdf',
                format='pdf')

# %% Display paths built from optimal transport with different values of 
# reference couplings and large epsilon

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

for i in range(n_sample):
   ax.plot(time_grid, 
            path_storage_EOTindep[i,:,0],
            path_storage_EOTindep[i,:,1])
    

ax.set_xlabel("time")
ax.set_ylabel("x")
ax.set_zlabel("y")

ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])


ax.set_title("Independent reference "+
             r"and $\varepsilon$ = {}".format(eps))
plt.savefig('Figures_EOTReference/Path_3D_Indepref_Larg_Eps.pdf', 
            format='pdf')


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

for i in range(n_sample):
    ax.plot(time_grid, path_storage_EOT_prior[i,:,0],
            path_storage_EOT_prior[i,:,1])

ax.set_xlabel("time")
ax.set_ylabel("x")
ax.set_zlabel("y")

ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
ax.set_title("Well-specified "+
             r"reference and $\varepsilon$ = {}".format(eps))
plt.savefig('Figures_EOTReference/Path_3D_Wellspec_Larg_Eps.pdf',
            format='pdf')


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
for i in range(n_sample):
    ax.plot(time_grid, path_EOT_missprior[i,:,0],
            path_EOT_missprior[i,:,1])

ax.set_xlabel("time")
ax.set_ylabel("x")
ax.set_zlabel("y")

ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
ax.set_title("Misspecified "+
             r"reference and $\varepsilon$ = {}".format(eps))
plt.savefig('Figures_EOTReference/Path_3D_Misspec_Larg_Eps.pdf', 
            format='pdf')


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

for i in range(n_sample):
   ax.plot(time_grid, 
            path_storage_otnoreg[i,:,0],
            path_storage_otnoreg[i,:,1])
    

ax.set_xlabel("time")
ax.set_ylabel("x")
ax.set_zlabel("y")

ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
ax.set_title("Unregularized optimal transport")
plt.savefig('Figures_EOTReference/Path_3D_UnregOT.pdf',
            format='pdf')


# %% Display paths built from entropic OT with different reference couplings
# and different values of epsilon

for it, vareps in enumerate(eps_grid):
    
    path_EOTindep = path_EOTindep_vareps[it]
    path_EOTprior = path_EOTprior_vareps[it]
    path_EOTmissprior = path_EOTmissprior_vareps[it]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    for i in range(n_sample):
       ax.plot(time_grid, 
                path_EOTindep[i,:,0],
                path_EOTindep[i,:,1])
       
    ax.set_xlabel("time")
    ax.set_ylabel("x")
    ax.set_zlabel("y")

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    ax.set_title(r"Independent reference and $\varepsilon$ = {}".format(vareps))
    
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    for i in range(n_sample):
        ax.plot(time_grid, path_EOTprior[i,:,0],
                path_EOTprior[i,:,1])
    
    ax.set_xlabel("time")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    ax.set_title("Well-specified reference "+
                 r"and $\varepsilon$ = {}".format(vareps))
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    for i in range(n_sample):
        ax.plot(time_grid, path_EOTmissprior[i,:,0],
                path_EOTmissprior[i,:,1])
    
    ax.set_xlabel("time")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    ax.set_title("Misspecified reference "+
                  r"and $\varepsilon$ = {}".format(vareps))



#%%

#2 dimensional representation

fig, ax = plt.subplots(figsize=(10, 8))
for i in range(n_sample):
    ax.plot(path_storage[i,:,0], path_storage[i,:,1])
    ax.set_xlim([-3, 3])
    ax.set_xlim([-3, 3])

ax.set_title("Figures_EOTReference/Paths from the diffusion", size=20)

    

fig, ax = plt.subplots(figsize=(10, 8))
for i in range(n_sample):
    ax.plot(path_storage_EOTindep[i,:,0], path_storage_EOTindep[i,:,1])
    ax.set_xlim([-3, 3])
    ax.set_xlim([-3, 3])

ax.set_title(r"Independent reference and $\varepsilon = {}$".format(eps),
             size=20)


fig, ax = plt.subplots(figsize=(10, 8))
for i in range(n_sample):
    ax.plot(path_storage_EOT_prior[i,:,0], path_storage_EOT_prior[i,:,1])
    ax.set_xlim([-3, 3])
    ax.set_xlim([-3, 3])

ax.set_title(r"Figures_EOTReference/Well-specified reference and $\varepsilon = {}$".format(eps),
             size=20)



fig, ax = plt.subplots(figsize=(10, 8))
for i in range(n_sample):
    ax.plot(path_storage_otnoreg[i,:,0], path_storage_otnoreg[i,:,1])
    ax.set_xlim([-3, 3])
    ax.set_xlim([-3, 3])

ax.set_title("Unregularized optimal transport", size=20)

#%%
fig = plt.figure(figsize=(18,5))
plt.suptitle("Independent reference - Number of margins = {}".format(n_times),
             size=20)
for it, vareps in enumerate(eps_grid):
    path = path_EOTindep_vareps[it]
    plt.subplot(1,3,it+1)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.title(r"$\varepsilon = {}$".format(vareps), size=14)
    for i in range(n_sample):
        plt.plot(path[i,:,0], path[i,:,1])
plt.savefig('Figures_EOTReference/Indepcoupling_sampling_2dim_proj.pdf', 
            format='pdf')


fig = plt.figure(figsize=(18,5))
plt.suptitle("Well-specified reference - Number of margins = {}".format(n_times),
             size=20)
for it, vareps in enumerate(eps_grid):
    path = path_EOTprior_vareps[it]
    plt.subplot(1,3,it+1)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.title(r"$\varepsilon = {}$".format(vareps), size=14)
    for i in range(n_sample):
        plt.plot(path[i,:,0], path[i,:,1])
plt.savefig('Figures_EOTReference/Goodrefe_sampling_2dim_proj.pdf',
            format='pdf')
      

fig = plt.figure(figsize=(18,5))
plt.suptitle("Misspecified reference - Number of margins = {}".format(n_times),
             size=20)
for it, vareps in enumerate(eps_grid):
    path = path_EOTmissprior_vareps[it]
    plt.subplot(1,3,it+1)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.title(r"$\varepsilon = {}$".format(vareps), size=14)
    for i in range(n_sample):
        plt.plot(path[i,:,0], path[i,:,1])
plt.savefig('Figures_EOTReference/Misspec_refe_sampling_2dim_proj.pdf',
            format='pdf')


#%% Impact of time discretization for independent and misspecified reference

eps_time = 0.1
n_marg_grid = [100,500,1000]
sample_indep_time = []
sample_missp_time = []

for it, n_marg in enumerate(n_marg_grid):
    #times where the marginals are available
    loc_time_grid = np.linspace(0.1, 1, num=n_marg)
    
    #true observed marginal covariances
    marginal_grid = grid_marge_var(t_grid=loc_time_grid, drift_mat= Drift, 
                   diff_mat=Diffu, init_mat= init_var)
    
    eot_prod_storage = []
    eot_refe_storage = []
    
    misspec_prior_grid = grid_ref_cov_OU(loc_time_grid, 
                                      drift_mat=Drift_prior, 
                                      diff_mat=Diffu_prior, 
                                      init_mat=init_var_prior)
    
    for it in range(n_marg-1):
        A_0 = marginal_grid[it]
        A_1 = marginal_grid[it+1]

        #Entropic OT with independent couplings
        
        eot_prod = cov_eot_prior(A = A_0, B = A_1, reg=eps_time) 
        eot_prod_storage.append(eot_prod)
    
        #Entropic OT coupling with misspecified reference
    
        missprior_cov = misspec_prior_grid[it]
        eot_missprior = cov_eot_prior(A = A_0, B = A_1, reg=eps_time, 
                                  Sigma = missprior_cov)
        eot_refe_storage.append(eot_missprior)

    #Sampling and storing
    
    path_eot_prod = sample_markov_bicoupling(loc_time_grid,
                                              eot_prod_storage,
                                              size_sample=n_sample)
    sample_indep_time.append(path_eot_prod)
    
    path_eotmisspec = sample_markov_bicoupling(loc_time_grid,
                                              eot_refe_storage,
                                              size_sample=n_sample)
    sample_missp_time.append(path_eotmisspec)
    
#%%

fig = plt.figure(figsize=(18,5))
plt.suptitle(r"Independent reference and $\varepsilon={}$".format(eps_time),
             size=20)
for it, n_marg in enumerate(n_marg_grid):
    path = sample_indep_time[it]
    plt.subplot(1,3,it+1)
    plt.title("Number of margins = {}".format(n_marg), size=14)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    for i in range(n_sample):
        plt.plot(path[i,:,0], path[i,:,1])
    plt.savefig('Figures_EOTReference/2DPath_indep_ref_timediscr.pdf', 
                format='pdf')


fig = plt.figure(figsize=(18,5))
plt.suptitle(r"Misspecified reference and $\varepsilon={}$".format(eps_time), 
             size=20)
for it, n_marg in enumerate(n_marg_grid):
    path = sample_missp_time[it]
    plt.subplot(1,3,it+1)
    plt.title("Number of margins = {}".format(n_marg), size=14)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    for i in range(n_sample):
        plt.plot(path[i,:,0], path[i,:,1])
    plt.savefig('Figures_EOTReference/2DPath_Misspec_ref_timediscr.pdf', 
                format='pdf')
