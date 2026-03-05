#genetrate staggered grids for 1D FV schemes

import numpy as np
import finite_volume.startup as strt
def generate_grid():
    """generate_grid Generate a 1D grid with `x_num` elements of size `x_size`.

    Parameters
    ----------
    x_size : float
    x_num : int

    Returns
    -------
    list
        Mesh
    """
    #x_grid = []  # the grid of spcae discretization
    x_size = strt.x_num - 1
    x_grid = np.array([(i + 0.5) * (1.0 / strt.x_num) for i in range(0, x_size)])

    return x_grid

"""Function populating the ghost cells for periodic boundary condition"""
bdary = fv.boundary_condition.per_bd
num_ghost = case.ng #number of ghost cells on each side
w_0 = bdary(w_0, num_ghost)
rho_0 = bdary(rho_0, num_ghost)
#Effective velocity part of the numerical flux on the interfaces including external edges
F_ev = np.array([(fv.convective_flux.flx_upwind(rho_0[i], rho_0[i+1],w_0[i])) for i in range(0,N+1)])
#Drift velocity part of the numerical flux on the interfaces including external edges
F_dv = np.array([(rho_0[i+1] - rho_0[i])/cell_size for i in range(0,N+1)])
#Flux = F_ev - kappa * nu * F_dv
Flx = np.array([(F_ev[i] - kappa * nu * F_dv[i]) for i in range(0,N+1)])
#Matrix blocks corresponding to the coupled linear system for w-v:
#-----------------------------------------------------------------
#w-update: W: coeffs of w, V: coeffs of v
W = np.zeros(shape=(N-1, N-1))
V = np.zeros(shape=(N-1, N-1))
for i in range(0,N-1):
    for j in range(0,N-1):
        if j==i:
            W[i][j] += 1.0 + lda * (1.0/4.0) * (Flx[i+2] - Flx[i]) + lda * ((1.0 - kappa) * nu/cell_size) * (rho_0[i+2] + rho_0[i+1])
            V[i][j] += -lda * ((1.0 - kappa) * nu * nu * kappa/cell_size) * (rho_0[i+2] + rho_0[i+1])
        elif j == i+1:
            W[i][j] += lda * (1.0/4.0) * (Flx[i+2] + Flx[i+1]) - lda * ((1.0 - kappa) * nu/cell_size) * rho_0[i+2]
            V[i][j] += lda * ((1.0 - kappa) * nu * nu * kappa/cell_size) * rho_0[i+2]
        elif j == i-1:
            W[i][j] += - lda * (1.0/4.0) * (Flx[i+1] + Flx[i]) - lda * ((1.0 - kappa) * nu/cell_size) * rho_0[i+1]
            V[i][j] += lda * ((1.0 - kappa) * nu * nu * kappa/cell_size) * rho_0[i+1]
#v-update: W_1: coeffs of v, V_1: coeffs of w
W_1 = np.zeros(shape=(N-1, N-1))
V_1 = np.zeros(shape=(N-1, N-1))
for i in range(0,N-1):
    for j in range(0,N-1):
        if j==i:
            W_1[i][j] += 1.0 + lda * (1.0/4.0) * (Flx[i+2] - Flx[i]) + lda * (kappa * nu/cell_size) * (rho_0[i+2] + rho_0[i+1])
            V_1[i][j] += -lda * (1.0/cell_size) * (rho_0[i+2] + rho_0[i+1])
        elif j == i+1:
            W_1[i][j] += lda * (1.0/4.0) * (Flx[i+2] + Flx[i+1]) - lda * (kappa * nu/cell_size) * rho_0[i+2]
            V_1[i][j] += lda * (1.0/cell_size) * rho_0[i+2]
        elif j == i-1:
            W_1[i][j] += - lda * (1.0/4.0) * (Flx[i+1] + Flx[i]) - lda * (kappa * nu/cell_size) * rho_0[i+1]
            V_1[i][j] += lda * (1.0/cell_size) * rho_0[i+1]
matrix_lhs = np.array(([W,V], [V_1,W_1]))
