# Description of the PDE being solved

#%% Demo for using an object
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi
import scipy.sparse as sparse
import scipy.sparse.linalg as spm

import finite_volume.finite_volume as fv

# Option 1: Load user input
# user_input = read('some_file.txt')

# Option 2: input is in the demo script
#Positive and negative parts of a real number
def pos(a):
    b = 0.5 * (math.fabs(a) + a)
    return b
def neg(a):
    b = 0.5 * (math.fabs(a) - a)
    return b
tf = 2.0
kappa = 0.5
nu = 0.1
gamma = 1.4
initial_condition = fv.initial_condition.disp_Riemann
case = fv.computational_case(a =-20.0, b = 20.0, Tf = 1.0, N = 50, dt = 0.00001, ng = 1)
"-------initialization of the scheme--------------"
a = case.a
b = case.b
N = case.N
l = b - a #length of the domain
cell_size = l/N #uniform cell size
dt = case.dt
lda = dt/cell_size

#Discretize the initial density by taking cell averages on PRIMAL CELLS
prim_edge = np.array([(a + i * cell_size) for i in range(0, N+1)])#edges of N uniform subintervals of (a,b)/edges of the primal cells including bdary a,b
x_prim = np.array([(prim_edge[i] + 0.5 * cell_size) for i in range(0, N)])#primal cell centres
rho_init = np.array([spi.quad(lambda x: (1.0/cell_size) * initial_condition(x)[0], prim_edge[i], prim_edge[i+1])[0] for i in range(0,N)])
#Discretize the initial velocity by taking cell averages on DUAL CELLS
x_dual = np.array([(prim_edge[i+1]) for i in range(0, N-1)]) #dual cell centres/internal edges/primal cell edges lying inside (a,b) hence excluding a,b
u_0 = np.array([spi.quad(lambda x: (1.0/cell_size) * initial_condition(x)[1], x_prim[i], x_prim[i+1])[1] for i in range(0,N-1)])
#Compute the DISCRETE x-DERIVATIVE of the initial density (initial DRIFT VELOCITY) on DUAL CELLS
Dx_rho_init = np.array([((rho_init[i+1]-rho_init[i])/cell_size) for i in range(0,N-1)])
#Compute the discrete EFFECTIVE VELOCITY on DUAL CELLS
w_0 = np.array([u_0[i] - kappa * nu * Dx_rho_init[i] for i in range(0,N-1)])
#----------------------------------plot discretized initial data--------------------------------------------------
#x = np.linspace(0, 1, num=int(1e2))
#rho0 = initial_condition(x)[1]
f, ax = plt.subplots(layout="constrained")
ax.plot(x_prim, rho_init, label=r"$\rho$")
ax.plot(x_dual, u_0, label=r"$u_0$")
ax.plot(x_dual, Dx_rho_init, label=r"$\partial_x \rho$")
ax.plot(x_dual, w_0, label=r"$w_0$")
ax.set_xlabel("x")
ax.set_title("Initial condition")
ax.legend()
#------------------------------------------------------------------------------------------------------------------
"""implementing periodic boundary condition for the initial data; populating the ghost cells"""
bdary = fv.boundary_condition.per_bd
num_ghost = case.ng #number of ghost cells on each side
w_0 = bdary(w_0, num_ghost)
#-------------------------------------------------------------------------------------------------------------------
"""-----------------------Update steps---------------------"""
"""Compute rho^0 (PRIMAL CELLS): solve a linear system"""
f_up = fv.convective_flux.flx_upwind
#-------------Entries of the sparse (M-)matrix A corresponding to the update for rho^0------------------------------
A = np.zeros(shape=(N,N))
for i in range(0,N):
    for j in range(0,N):
        if j==i:
            A[i][j] += 1.0 + lda * (pos(w_0[i+1]) + neg(w_0[i])) + kappa * nu * (2.0/cell_size) * lda
        elif j == i+1:
            A[i][j] += - lda * neg(w_0[i+1]) - (kappa * nu * lda)/cell_size
        elif j == i-1:
            A[i][j] += - lda * pos(w_0[i]) - (kappa * nu * lda)/cell_size
A[0][N-1] += - lda * pos(w_0[N]) - (kappa * nu * lda)/cell_size
A[N-1][0] += - lda * neg(w_0[0]) - (kappa * nu * lda)/cell_size
#--------------------------------------------------------------------------------------------------------------------
#------------Solving for rho^0 from the corresponding linear problem--------------------------------------------------
rho_0 = spm.spsolve(A, rho_init)
ax.plot(x_prim, rho_0, label=r"$\rho^0$")
ax.legend()
#-------------Time loop for updates------------------------
#Compute dual average of the discrete mass on the DUAL CELLS
rho_init_d = np.array([(0.5 * (rho_init[i+1]+rho_init[i])) for i in range(0,N-1)])
rho_0_d = np.array([(0.5 * (rho_0[i+1]+rho_0[i])) for i in range(0,N-1)])
#Pressure scaling step: compute the scaled pressure gradient on the DUAL CELLS
sc_pr_grad = np.array([(math.sqrt(rho_0_d[i]/rho_init_d[i]) * (rho_0[i+1]**gamma - rho_0[i]**gamma)/cell_size) for i in range(0,N-1)])
ax.plot(x_dual, sc_pr_grad, label=r"scaled pressure")
ax.legend()
#Prediction step: solve a linear system to get the intermediate effective vel. and the drift vel.






#%%
