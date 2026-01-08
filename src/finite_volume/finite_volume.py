# Example of what finite_volume could look like
#%%
import numpy as np
import scipy.integrate as spi


class initial_condition:
    """Library of initial conditions."""

    @staticmethod
    def disp_Riemann(x):
        """Dispersive Riemann problem from (Calgaro et al, 2024)"""
        rho0 = 1.5 - 0.5 * np.tanh(x/0.1)
        u0 = np.zeros_like(x)
        return (rho0,u0)


class computational_case:
    """Describes a computational case.
    Contains data defining the domain and discretization parameters"""

    def __init__(self, a = 0.0, b = 1.0, Tf = 1.0, N = 100, dt=1e-4):
        """Constructor for computational_case class.

        Parameters
        ----------
        N : int,
            number of control volumes.
        Tf: float,
            final time.
        a, b: float a<b, the interval (a,b) is the domain.
        dt: float, time-step size
        """
        self.a = a
        self.b = b
        self.Tf = Tf
        self.N = N
        self.dt = dt


def discretization(case, data):
    """Perform the computation for the given case.
    Parameters
    ----------
    computational_case: parametrs/object specific
                        to the computational case
    initial_condition: initial condition under consiaderation"""
    

# %%
