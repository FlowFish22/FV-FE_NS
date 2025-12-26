#startup user input
import math
module = int(input('Which module?:\n' 
                ' 0 = 1D linear advenctoin'))
if module == 0:
    x_num = int(input('\n Number of mesh cells:'))
    speed = float(input('\n Enter a constant value for advection:'))
else:
    print('NA')
    quit()
init_data = int(input('\n Enter initial data:\n'
                       ' 0 = Riemann problem'))
if init_data == 0:
    x = float(input('\n Enter point of discontinuity in [0,1]:'))
else:
    print('NA')
    quit()
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
