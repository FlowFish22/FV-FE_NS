# Description of the PDE being solved

# Demo for using an object
import matplotlib.pyplot as plt

import finite_volume

# Option 1: Load user input
# user_input = read('some_file.txt')

# Option 2: input is in the demo script
ul, ur = 1.0, 2.0, tf = 2.0
initial_condition = finite_volume.initial_condition.shocktube
case = finite_volume.computation_case(
    ul=ul, ur=ur, tf=tf, initial_condition=initial_condition
)
results = finite_volume.discretization(case)

# Plot the result
u = results.u
x = results.x
plt.plot(x, u)
