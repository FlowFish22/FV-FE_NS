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
case = fv.computational_case(N = 3)
N = case.N
#Discretize the initial density by taking cell averages on primal cells
x_prim = np.array([((i + 0.5) * (1.0 /N)) for i in range(0, N)])
b = spi.quad(np.sin, 0, 1)
r_avg = np.array([spi.quad(lambda x: x, i/N, (i+1.0)/N)[0] for i in range(0,N)])

#----------------------------------
#x = np.linspace(0, 1, num=int(1e2))
#rho0 = initial_condition(x)[1]
#f, ax = plt.subplots(layout="constrained")
#ax.plot(x, rho0, label=r"$\rho$")
#ax.plot(x, u0, label="u")
#ax.set_xlabel("x")
#ax.set_title("Initial condition")
#ax.legend()

# Plot the result
# u = results.u
# x = results.x
# plt.plot(x, u)



# %%
