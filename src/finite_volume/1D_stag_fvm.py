"""Some functions related to finite volume."""

import numpy as np
import finite_volume.startup as strt
def generate_grid():
    """generate_grid Generate a 1D grid with `x_num` elements of size `x_size`.

    Parameters
    ----------
    x_size : float
    x_num : int

    Returns
    -------
    list
        Mesh
    """
    #x_grid = []  # the grid of spcae discretization
    x_size = strt.x_num - 1
    x_grid = np.array([(i + 0.5) * (1.0 / strt.x_num) for i in range(0, x_size)])

    return x_grid
