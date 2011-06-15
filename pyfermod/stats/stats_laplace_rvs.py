"""
Returns the array of random variates
for the Laplace probability distribution
using the given arrays for the abscissa or template
values and each of the parameters values.
"""
import numpy
import pyferret
import pyferret.stats

DISTRIB_NAME = "Laplace"
FUNC_NAME = "rvs"


def ferret_init(id):
    """
    Initialization for the stats_laplace_rvs Ferret PyEF
    """
    return pyferret.stats.getinitdict(DISTRIB_NAME, FUNC_NAME)


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Result array assignment for the stats_laplace_rvs Ferret PyEF
    """
    pyferret.stats.assignresultsarray(DISTRIB_NAME, FUNC_NAME,
                                      result, resbdf, inputs, inpbdfs)

