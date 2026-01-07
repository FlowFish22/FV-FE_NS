#initialize the pressure correction scheme
import math
import numpy as np
import scipy.integrate as integrate

import finite_volume.startup as strt
import finite_volume.bdary as bnd
import finite_volume.initial_condn as incon
import finite_volume.stag_grid_1D as grid

#obtain \rho^{-1} from as cell averages from the initial density



