# Description of the PDE being solved

#%% Demo for using an object
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi
import scipy.sparse.linalg as spm
from scipy.sparse import coo_array, bmat

import finite_volume.finite_volume as fv

# Option 1: Load user input
# user_input = read('some_file.txt')

# Option 2: input is in the demo script
#Positive and negative parts of a real number
def pos(a):
    return np.maximum(a,0)
def neg(a):
    return np.minimum(a,0)
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
lda2 = dt/(cell_size * cell_size)
c = (kappa * nu * lda)/cell_size

#Discretize the initial density by taking cell averages on PRIMAL CELLS
prim_edge = np.array([(a + i * cell_size) for i in range(0, N+1)])#edges of N uniform subintervals of (a,b)/edges of the primal cells including bdary a,b
x_prim = np.array([(prim_edge[i] + 0.5 * cell_size) for i in range(0, N)])#primal cell centres
rho_init = np.array([spi.quad(lambda x: (1.0/cell_size) * initial_condition(x)[0], prim_edge[i], prim_edge[i+1])[0] for i in range(0,N)])
#Discretize the initial velocity by taking cell averages on DUAL CELLS
x_dual = np.array([(prim_edge[i+1]) for i in range(0, N-1)]) #dual cell centres/internal edges/primal cell edges lying inside (a,b) hence excluding a,b
u_0 = np.array([spi.quad(lambda x: (1.0/cell_size) * initial_condition(x)[1], x_prim[i], x_prim[i+1])[1] for i in range(0,N-1)])
#Compute the initial DRIFT VELOCITY) on DUAL CELLS
v_init = np.array([((rho_init[i+1]- rho_init[i])/(cell_size * 0.5 * (rho_init[i+1] + rho_init[i]))) for i in range(0,N-1)])
#Compute the discrete EFFECTIVE VELOCITY on DUAL CELLS
w_0 = np.array([u_0[i] - kappa * nu * v_init[i] for i in range(0,N-1)])
#----------------------------------plot discretized initial data--------------------------------------------------
#x = np.linspace(0, 1, num=int(1e2))
#rho0 = initial_condition(x)[1]
f, ax = plt.subplots(layout="constrained")
ax.plot(x_prim, rho_init, label=r"$\rho$")
ax.plot(x_dual, u_0, label=r"$u_0$")
ax.plot(x_dual, v_init, label=r"$\partial_x \rho$")
ax.plot(x_dual, w_0, label=r"$w_0$")
ax.set_xlabel("x")
ax.set_title("Initial condition")
ax.legend()
#-------------------------------------------------------------------------------------------------------------------
"""-----------------------Update steps---------------------"""
"""Compute rho^0 (PRIMAL CELLS): solve a linear system"""
f_up = fv.convective_flux.flx_upwind
#-------------Entries of the sparse (M-)matrix A corresponding to the update for rho^0------------------------------
p_linsolv = fv.solver_assembly.primal_linsolv_periodic
A = p_linsolv(w_0, lda, c, neg, pos)
#--------------------------------------------------------------------------------------------------------------------
#------------Solving for rho^0 from the corresponding linear problem--------------------------------------------------
rho_0 = spm.spsolve(A, rho_init)
ax.plot(x_prim, rho_0, label=r"$\rho^0$")
#ax.legend()
#------------------------------------------------------------------------------------------------------------------
#Compute dual average of the discrete mass on the DUAL CELLS
rho_init_d = np.array([(0.5 * (rho_init[i+1]+rho_init[i])) for i in range(0,N-1)])
rho_0_d = np.array([(0.5 * (rho_0[i+1]+rho_0[i])) for i in range(0,N-1)])
#Pressure scaling step: compute the scaled pressure gradient on the DUAL CELLS
sc_pr_grad = np.array([(math.sqrt(rho_0_d[i]/rho_init_d[i]) * (rho_0[i+1]**gamma - rho_0[i]**gamma)/cell_size) for i in range(0,N-1)])
ax.plot(x_dual, sc_pr_grad, label=r"scaled pressure")                                                                                                                             
ax.legend()
#Prediction step: solve a linear system to get the intermediate effective vel. and the drift vel.
#------------------------------------------------------------------------------------------------
#Effective velocity part of the numerical flux on the interfaces excluding external edges
f_ev = np.array([(fv.convective_flux.flx_upwind(rho_0[i], rho_0[i+1],w_0[i])) for i in range(0,N-1)])
#Drift velocity part of the numerical flux on the interfaces excluding external edges
f_dv = np.array([(rho_0[i+1] - rho_0[i])/cell_size for i in range(0,N-1)])
#Flux = F_ev - kappa * nu * F_dv
flx = np.array([(f_ev[i] - kappa * nu * f_dv[i]) for i in range(0,N-1)])
c1 = lda * (1.0/4.0)
c2 = nu * (1.0 - kappa) * lda2
c3 = kappa * nu * lda2
d = kappa * nu * nu * (1 - kappa) * lda2
d_linsolv = fv.solver_assembly.dual_linsolv
d_linsolv_dif = fv.solver_assembly.dual_linsolv_dif
build_mtx = fv.solver_assembly.build_matrix

"""Matrix blocks corresponding to the linear system for solving tilde{w} and v"""
W1 = d_linsolv(flx, rho_0, c1, c2) #tilde{w} part of tilde{w} eqn
V1 = d_linsolv_dif(rho_0, d) #v part of tilde{w} eqn
V2 = d_linsolv(flx, rho_0, c1, c3) #v part of v eqn
W2 = d_linsolv_dif(rho_0, lda2) #tilde{w} part of w eqn

M = build_mtx(W1,V1, W2, V2)

"""Compute the intermediate effective velocity and the drift velocity"""
rhs_tw = rho_init_d * w_0 - sc_pr_grad #rhs of the w equation
rhs_v = rho_init_d * v_init #rhs of the v equation
rhs_dual = np.concatenate((rhs_tw, rhs_v)) #build the vector on right hand side
twv = spm.spsolve(M, rhs_dual) #vector (tw, v)
tw, v = twv[:len(twv)//2], twv[len(twv)//2:]

ax.plot(x_dual, tw, label=r"$\tilde{w}$ first update")
#ax.plot(x_dual, v, label=r"$v$ first update")
ax.legend()

#%%
