# Description of the PDE being solved

#%% Demo for using an object
import matplotlib.pyplot as plt
import numpy as np

import finite_volume.finite_volume as fv

# Option 1: Load user input
# user_input = read('some_file.txt')

# Option 2: input is in the demo script
tf = 2.0
initial_condition = fv.initial_condition.disp_Riemann
case = fv.computational_case(N = 1000)
#results = fv.discretization(case)

#Discretize initial condition
N = case.N

# Plot the initial condition
x = np.linspace(0, 1, num=int(1e2))
(u0, rho0) = initial_condition(x)
f, ax = plt.subplots(layout="constrained")
ax.plot(x, u0, label=r"$u$")
ax.plot(x, rho0, label=r"$\rho$")
ax.set_xlabel("x")
ax.set_title("Initial condition")
ax.legend()

# Plot the result
# u = results.u
# x = results.x
# plt.plot(x, u)



# %%
