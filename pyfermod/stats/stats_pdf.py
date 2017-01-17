"""
Returns the array of probability density function values for a
continuous probability distribution and set of abscissa values.
"""

from __future__ import print_function

import numpy
import scipy.stats
import pyferret
import pyferret.stats


def ferret_init(id):
    """
    Initialization for the stats_pdf python-backed ferret external function
    """
    axes_values = [ pyferret.AXIS_IMPLIED_BY_ARGS ] * pyferret.MAX_FERRET_NDIM
    true_influences = [ True ] * pyferret.MAX_FERRET_NDIM
    false_influences = [ False ] * pyferret.MAX_FERRET_NDIM
    retdict = { "numargs": 3,
                "descript": "Returns probability density function values for a continuous probability distribution",
                "axes": axes_values,
                "argnames": ("PTS", "PDNAME", "PDPARAMS"),
                "argdescripts": ("Points at which to calculate the probability density function values",
                                 "Name of a continuous probability distribution",
                                 "Parameters for this continuous probability distribution"),
                "argtypes": (pyferret.FLOAT_ARRAY, pyferret.STRING_ONEVAL, pyferret.FLOAT_ARRAY),
                "influences": (true_influences, false_influences, false_influences),
              }
    return retdict


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Assigns result with the probability density function values for the
    continuous probability distribution indicated by inputs[1] (a string)
    using the parameters given in inputs[2] at the abscissa values given
    by inputs[0].  For undefined abscissa values, the result value will
    be undefined.
    """
    distribname = inputs[1]
    distribparams = inputs[2].reshape(-1)
    distrib = pyferret.stats.getdistrib(distribname, distribparams)
    badmask = ( numpy.fabs(inputs[0] - inpbdfs[0]) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(inputs[0]))
    goodmask = numpy.logical_not(badmask)
    result[badmask] = resbdf
    # array[goodmask] is a flattened array
    result[goodmask] = distrib.pdf(inputs[0][goodmask])


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # make sure ferret_init does not have problems
    info = ferret_init(0)

    # Normal distribution along the Y axis
    dimen = 25
    mu = 5.0
    sigma = 2.0
    distf = scipy.stats.norm(mu, sigma)
    xvals = numpy.linspace(mu - 2.5 * sigma, mu + 2.5 * sigma, dimen)
    pdfvals = distf.pdf(xvals)

    pfname = "norm"
    pfparams = numpy.array([mu, sigma], dtype=numpy.float64)
    inpbdfs = numpy.array([-1.0, 0.0, 0.0], dtype=numpy.float64)
    resbdf = numpy.array([-2.0], dtype=numpy.float64)
    abscissa = numpy.empty((1, dimen, 1, 1, 1, 1), dtype=numpy.float64, order='F')
    expected = numpy.empty((1, dimen, 1, 1, 1, 1), dtype=numpy.float64, order='F')
    for j in range(dimen):
        if (j % 7) == 3:
            abscissa[0, j, 0, 0, 0, 0] = inpbdfs[0]
            expected[0, j, 0, 0, 0, 0] = resbdf[0]
        else:
            abscissa[0, j, 0, 0, 0, 0] = xvals[j]
            expected[0, j, 0, 0, 0, 0] = pdfvals[j]
    result = -888.0 * numpy.ones((1, dimen, 1, 1, 1, 1), dtype=numpy.float64, order='F')
    ferret_compute(0, result, resbdf, (abscissa, pfname, pfparams), inpbdfs)
    if not numpy.allclose(result, expected):
        print("Expected (flattened) = %s" % str(expected.reshape(-1)))
        print("Result (flattened) = %s" % str(result.reshape(-1)))
        raise ValueError("Unexpected result")

    # All successful
    print("Success")

