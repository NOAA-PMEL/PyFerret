"""
Returns the array of probability distribution function values
for the Exponential probability distribution
using the given arrays for the abscissa or template
values and each of the parameters values.
"""
import numpy
import pyferret
import pyferret.stats

DISTRIB_NAME = "Exponential"
FUNC_NAME = "pdf"


def ferret_init(id):
    """
    Initialization for the stats_expon_pdf Ferret PyEF
    """
    return pyferret.stats.getinitdict(DISTRIB_NAME, FUNC_NAME)


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Result array assignment for the stats_expon_pdf Ferret PyEF
    """
    pyferret.stats.assignresultsarray(DISTRIB_NAME, FUNC_NAME,
                                      result, resbdf, inputs, inpbdfs)

