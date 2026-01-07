#startup user input
import math
#ask user which scheme to be implemented
module = int(input('Which module?:\n' 
                ' 0 = 1D aug Navier-Stokes pressure-correction'))
#ask user to choose the initial data
init_data = int(input('\n Enter initial data:\n'
                       ' 0 = Dispersive Riemann problem'))
if module == 0:
    x_num = int(input('\n Number of mesh cells:'))
    T = float(input('\n Final time:'))
    dt = float(input('\n Time-step size:'))
    nu =float(input('\n Viscous coeff:'))
else:
    print('NA')
    quit()
#ask user the boundary condition to be implemented
bdary = int(input('\nWhich bdary condn?:'
                  '\n 0 = Periodic'
                  '\n 1= other'))
if bdary != 0:
    ls_condn = int(input('\n Left boundary condn?'
                                 '\n 0 = Dirichlet'
                                 '\n 1 = Extrapolation'
                                 '\n 2 = None'))
    rs_condn = int(input('\n right boundary condn?'
                                 '\n 0 = Dirichlet'
                                 '\n 1 = Extrapolation'
                                 '\n 2 = None'))
    if ls_condn == 0:
        bd_val_left = float(input('\n Enter left boundary value:'))
    if rs_condn == 0:
        bd_val_right = float(input('\n Enter right boundary value:'))
   
else:
    print('NA')
    quit()
