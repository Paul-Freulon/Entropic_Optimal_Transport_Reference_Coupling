#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:34:30 2026

@author: freulon
"""

This repository contains the code associated to the paper 
"Entropic optimal transport beyond product reference couplings: the Gaussian case on Euclidean space" written by 
Paul Freulon, Nikitas Georgakis, and Victor Panaretos.
The preprint can be found at https://arxiv.org/abs/2507.01709


#### Organization of the repository

This repository is organized into three folders. 

The file named Functions_EOT_Ref_Couplings.py contains the functions to solve the 
entropic OT problem with reference couplings when the input measures are Gaussians.
In the script Functions_Gauss_Process_Sampling.py are the functions to sample from discrete-time
Gaussian processes. In the same sripts some functions compute couplings that can be chosen
as reference couplings for the entropic OT problems.

The Scripts folder contains the code to reproduce the experiments of the paper.
In Sampling_Gauss_Reference_EOT.py is the code where reference Gaussian processes are chosen
to be independent-coordinates with same distribution. In Sampling_mixsmooth_EOT.py, a reference 
process with independent coordinates but different distributions is chosen. This allows to 
generate paths where each coordinate has a different regularity. In Impact_CorParam_RealLine.py
is displayed the solution of the EOT when the input measures are one dimensional Gaussians.

All the figures of the paper are stored in the folder named Figures_EOTReference

#### How long does it take to run the scripts?

All scripts have been run on a personal computer.
The script "Sampling_mixsmooth_EOT.py" takes around two minutes to run. 
The main bottleneck is the approximation of the marginal distributions. 
The marginal covariances have explicit formulae involving integrals that we 
approximate by Riemman sums.
For 1000 marginals, solving the 999 entropic OT problems take less 
than one second. 
As more experiments are performed in "Sampling_Gauss_Reference_EOT.py" this 
script takes around eight minutes to run.
Running the script Impact_CorParam_RealLine.py is almost immediate.

#### Versions of the required libraries
The following libraries have been used.
-----
matplotlib                          3.8.2
numpy                               1.25.2
scipy                               1.9.3
session_info                        v1.0.1
-----
The following version of python and OS have been used.
-----
Python 3.10.12 (main, Mar  3 2026, 11:56:32) [GCC 11.4.0]
Linux-6.8.0-106-generic-x86_64-with-glibc2.35
-----