"""
Returns the array of probability mass function values
for the Poisson probability distribution using
the given arrays for the abscissa values and each of the parameters values.
"""
import numpy
import pyferret
import pyferret.stats

DISTRIB_NAME = "Poisson"
FUNC_NAME = "pmf"


def ferret_init(id):
    """
    Initialization for the stats_poisson_pmf Ferret PyEF
    """
    return pyferret.stats.getinitdict(DISTRIB_NAME, FUNC_NAME)


def ferret_custom_axes(id):
    """
    Custom axis definitions for the stats_poisson_pmf Ferret PyEF
    """
    return pyferret.stats.getcustomaxisvals(id, DISTRIB_NAME);


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Result array assignment for the stats_poisson_pmf Ferret PyEF
    """
    pyferret.stats.assignresultsarray(DISTRIB_NAME, FUNC_NAME,
                                      result, resbdf, inputs, inpbdfs)

