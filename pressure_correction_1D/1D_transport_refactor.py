# Description of the PDE being solved

#%% Demo for using an object
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi

import finite_volume.finite_volume as fv

# Option 1: Load user input
# user_input = read('some_file.txt')

# Option 2: input is in the demo script
tf = 2.0
initial_condition = fv.initial_condition.disp_Riemann
case = fv.computational_case(a =-20.0, b = 20.0, Tf = 1.0, N = 50, dt = 0.00001)
"-------initialization of the scheme--------------"
a = case.a
b = case.b
N = case.N
l = b - a #length of the domain
cell_size = l/N #uniform cell size

#Discretize the initial density by taking cell averages on primal cells
prim_edge = np.array([(a + i * cell_size) for i in range(0, N+1)])#edges of N uniform subintervals of (a,b)/edges of the primal cells including bdary a,b
x_prim = np.array([(prim_edge[i] + 0.5 * cell_size) for i in range(0, N)])#primal cell centres
rho_init = np.array([spi.quad(lambda x: (1.0/cell_size) * initial_condition(x)[0], prim_edge[i], prim_edge[i+1])[0] for i in range(0,N)])
#Discretize the initial velocity by taking cell averages on dual cells
x_dual = np.array([(prim_edge[i+1]) for i in range(0, N-1)]) #dual cell centres/internal edges/primal cell edges lying inside (a,b) hence excluding a,b
u_0 = np.array([spi.quad(lambda x: (1.0/cell_size) * initial_condition(x)[1], x_prim[i], x_prim[i+1])[1] for i in range(0,N-1)])
#----------------------------------plot discretized initial data--------------------------------------------------
#x = np.linspace(0, 1, num=int(1e2))
#rho0 = initial_condition(x)[1]
f, ax = plt.subplots(layout="constrained")
ax.plot(x_prim, rho_init, label=r"$\rho$")
ax.plot(x_dual, u_0, label="u")
ax.set_xlabel("x")
ax.set_title("Initial condition")
ax.legend()

# Plot the result
# u = results.u
# x = results.x
# plt.plot(x, u)



# %%
