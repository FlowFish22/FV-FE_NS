# Example of what finite_volume could look like
import numpy as np


class initial_condition:
    """Library of initial conditions."""

    @staticmethod
    def shock_tube(x):
        """Shock tube initial condition from (Toto et al, 2010)"""
        u0 = np.sin(x)
        v0 = np.cos(x)
        rho0 = np.tan(x)
        return (u0, v0, rho0)


class computational_case:
    """Describes a computational case."""

    def __init__(self, ul=1.0, name="default name"):
        """Constructor for computational_case class.

        Parameters
        ----------
        ul : float, optional
            Left value
        name : str, optional
            Name of the computational case.
        """
        self.ul = 1.0
        self.name = name

    def print_name(self):
        print(self.name)


def discretization(case: computational_case):
    """Perform the computation for the given case."""
    return case.ul
