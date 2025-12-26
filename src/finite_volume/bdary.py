import math
import numpy as np
import finite_volume.startup as strt
#currently either only left or only right or both sides with same boundary conditon is implemented
#periodic bdary condition
ls = strt.ls_condn
rs = strt.rs_condn
def per_bd(a):
    y = np.zeros(len(a) + 2) #one layers of ghost cell on each side
    y[-1] += a[0] #ghost cell with the right boudary take value from the first domain cell on the left
    y[0] += a[-1] #ghost cell with the left boundary take value from the last domain cell on the right
    y[1:-1] += a
    return y
def dir_bd(a, side_index):
    if side_index == 0: #left
        y = np.zeros(len(a) + 1) #adding one layer of ghost cell on one side
        y[0] += strt.bd_val_left
        y[1:] += a
    elif side_index == 1: #right
        y = np.zeros(len(a) + 1) #adding one layer of ghost cell on one side
        y[-1] += strt.bd_val_right
        y[:-1] += a
    elif side_index == 2: #both
        y = np.zeros(len(a) + 2) #adding one layer of ghost cells on both sides
        y[0] += strt.bd_val_left
        y[-1] += strt.bd_val_right
        y[1:-1] += a
    return y
def extrapl_bd(a, side_index):
    if side_index == 0: #left
        y = np.zeros(len(a) + 1) #adding one layer of ghost cell on one side
        y[0] += a[0]
        y[1:] += a
    elif side_index == 1: #right
        y = np.zeros(len(a) + 1) #adding one layer of ghost cell on one side
        y[-1] += a[-1]
        y[1:] += a
    elif side_index == 2: #both
        y = np.zeros(len(a) + 2) #adding one layer of ghost cells on both sides
        y[0] += a[0]
        y[-1] += a[-1]
        y[1:-1] += a
    return y

def bd(x):
    if strt.bdary == 0:
        y = per_bd(x)
    elif strt.bdary == 1:
        if strt.ls_condn == 2:
            y = dir_bd(x,1) if strt.rs_condn == 0 else extrapl_bd(x,1)
        if strt.rs_condn == 2:
            y = dir_bd(x,0) if strt.ls_condn == 0 else extrapl_bd(x,0)
        else:
            if strt.ls_condn == 0 and strt.rs_condn == 0:
                y = dir_bd(x,2)
            elif strt.ls_condn == 1 and strt.rs_condn == 1:
                y = extrapl_bd(x,2)
            else:
                print('NA')
                quit()
    return y
