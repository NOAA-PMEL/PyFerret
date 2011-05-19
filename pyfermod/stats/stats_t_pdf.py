"""
Returns the array of probability distribution function values
for the Students-T probability distribution using
the given arrays for the abscissa values and each of the parameters values.
"""
import numpy
import pyferret
import pyferret.stats

DISTRIB_NAME = "Students-T"
FUNC_NAME = "pdf"


def ferret_init(id):
    """
    Initialization for the stats_t_pdf Ferret PyEF
    """
    return pyferret.stats.getinitdict(DISTRIB_NAME, FUNC_NAME)


def ferret_custom_axes(id):
    """
    Custom axis definitions for the stats_t_pdf Ferret PyEF
    """
    return pyferret.stats.getcustomaxisvals(id, DISTRIB_NAME);


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Result array assignment for the stats_t_pdf Ferret PyEF
    """
    pyferret.stats.assignresultsarray(DISTRIB_NAME, FUNC_NAME,
                                      result, resbdf, inputs, inpbdfs)

