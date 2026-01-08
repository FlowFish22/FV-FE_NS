# A finite difference explicit upwind scheme for linear advection in 1D
import math

import numpy as np
from matplotlib import pyplot as plt

#import finite_volume.fvm as fvm
import finite_volume.startup as strt
import finite_volume.bdary as bnd

print("1D Linear transport")
a = strt.speed # Constant advection
#x = 0.5  # Point of discontinuity in [0,1]
u_L, u_R = 2.0, 1.0  # Left and right values
#x_size = 50  # Grid size
#x_num = x_size - 1
T = 0.75 # Final time
dt = 1.0/1000.0  # Time-step
dx = 1/strt.x_num
t_step_num = math.floor(T / dt)
x_grid = fvm.generate_grid()


# initialize the Riemann initial data
u_0 = np.where(x_grid < strt.x, u_L, u_R)
# implementing boundary condition
u = bnd.bd(u_0)
# time integration and upwind update
def f_up(u, v):
    f = a * (1.0/dx) * (u - v) #finite difference gradient
    return f
for n in range(1, t_step_num):
    u_old = u.copy()
    for i in range(1, strt.x_num - 1):
        Flx = f_up(u_old[i], u_old[i-1]) if a>=0 else f_up(u_old[i+1], u_old[i]) #upwind flux
        u[i] = u_old[i] - dt * Flx


#exact solution
u_exact = np.where(x_grid < strt.x + a * T, u_L, u_R)

if strt.ls_condn == 2 or strt.rs_condn == 2:
    if strt.ls_condn == 2:
        u_plt = u[:-1]
    if strt.rs_condn == 2:
        u_plt = u[1:]
else:
    u_plt = u[1:-1]

f, ax = plt.subplots(layout="constrained")
ax.plot(x_grid, u_plt, label="Computed", linestyle="-", color="b")
ax.plot(x_grid, u_exact, label="Exact", linestyle="--", color="r")
ax.plot(x_grid, u_0, label="initial", linestyle="-", color="g")
ax.set_title("1D linear advection")
f.legend()
