import numpy as np

n_sims = 1000
n_timesteps = 1000
n_spatial_dims = 3
n_nodes = 1
n_params = 1
n_totdims = n_spatial_dims*2*n_nodes + n_params

sigma = 10
rho = 28
beta = 8.0/3

sim_object = np.zeros(n_sims, n_timesteps, n_nodes, n_totdims)

def getXV_derivatives(xvals, vvals):
    """
    xvals and vvals are n_spatial_dims x 1 matrices representing the position and velocity, respectively
    Inputs: xvals, vvals: position and velocity coordinates at time t
    Outputs: vnew, anew: returns the new velocity and position at time t
    """

    vnew = np.zeros(n_spatial_dims)
    anew = np.zeros(n_spatial_dims)

    vnew[0] = sigma*(xvals[1]-xvals[0])
    vnew[1] = xvals[0]*(rho-xvals[2])-xvals[1]
    vnew[2] = xvals[0]*xvals[1] - beta*xvals[2]

    #no update to a_new, since velocity is not a independent variable for this system
    return vnew, anew