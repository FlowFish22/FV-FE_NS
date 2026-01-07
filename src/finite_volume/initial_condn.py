#Contains the initial data
import math
import numpy as np

import finite_volume.startup as strt
import finite_volume.bdary as bnd

if strt.init_data == 0:
    def rho_0(x):
        rho = 1.5 - 0.5 * np.tanh(x/0.2)
        return rho
    def u_0(x):
        u = 0
        return u 