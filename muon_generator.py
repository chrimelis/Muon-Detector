import random
import numpy as np
import muon_detector_helpers as md

###################################################
# Script used for creating randomly a specific
# instance of a Monte Carlo Simulation. 
# It generates positions and velocities of muons
# on the top sheet of the detector according to
# the modelling assumptions
###################################################

# The sheets of the detector are assumed to be square
a = 1   # the side of the square surface
# Generate uniformly N particles on the 1st sheet of the detector (top sheet)
N = 10**5
x_top_data = np.fromiter((random.uniform(0,a) for _ in range(N)), dtype=float) # x is uniformly distributed
y_top_data = np.fromiter((random.uniform(0,a) for _ in range(N)), dtype=float) # y is uniformly distributed
# Generate the velocity direction in 3D space
phi_data = np.fromiter((random.uniform(0,2*np.pi) for _ in range(N)), dtype=float)    # phi (the azimuthal angle) is uniformly distributed
# If θ' is the polar angle of spherical coordinates
# θ' = pi - theta, with theta taking values in [0,pi/2) according to the distribution theta = |v| where v ~ 2/pi * cos^2(v) with v in [-pi/2,pi/2]
# theta ~ cos^2 distribution
# Generate theta angles for all N particles
cos_sq = md.cosine_sq(a=0,b=np.pi/2) 
theta_data = cos_sq.rvs(size=N,random_state=None)

# Store the numpy arrays in an efficient format .npz
np.savez('data.npz', x_top_data=x_top_data, y_top_data=y_top_data, theta_data=theta_data, phi_data=phi_data)

