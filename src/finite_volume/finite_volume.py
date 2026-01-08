# Example of what finite_volume could look like
import numpy as np
import scipy.integrate as spi


class initial_condition:
    """Library of initial conditions."""

    @staticmethod
    def disp_Riemann(x):
        """Dispersive Riemann problem from (Calgaro et al, 2024)"""
        rho0 = 1.5 - 0.5 * np.tanh(x/0.2)
        u0 = np.zeros_like(x)
        return (rho0,u0)


class computational_case:
    """Describes a computational case."""

    def __init__(self, N):
        """Constructor for computational_case class.

        Parameters
        ----------
        N : float,
            number of control volumes.
        """
        self.N = N

def discretization(case, data):
    """Perform the computation for the given case.
    Parameters
    ----------
    computational_case: parametrs/object specific
                        to the computational case
    initial_condition: initial condition under consiaderation"""
    
