"""
Returns the array of probability density function values for a
discrete probability distribution and set of abscissa values.
"""
import sys
import numpy
import scipy.stats
import pyferret
import pyferret.stats


def ferret_init(id):
    """
    Initialization for the stats_pmf python-backed ferret external function
    """
    retdict = { "numargs": 3,
                "descript": "Returns probability mass function values for a discrete distribution",
                "axes": (pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS),
                "argnames": ("Abscissae", "PDName", "PDParams"),
                "argdescripts": ("Points at which to calculate the probability mass function values",
                                 "Name of a discrete probability distribution",
                                 "Parameters for this discrete probability distribution"),
                "argtypes": (pyferret.FLOAT_ARG, pyferret.STRING_ARG, pyferret.FLOAT_ARG),
                "influences": ((True,  True,  True,  True),
                               (False, False, False, False),
                               (False, False, False, False)),
              }
    return retdict


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Assigns result with the probability mass function values for the discrete
    probability distribution indicated by inputs[1] (a string) using the
    parameters given in inputs[2] at the abscissa values given by inputs[0].
    """
    distribname = inputs[1]
    distribparams = inputs[2].reshape(-1)
    distrib = pyferret.stats.getdistrib(distribname, distribparams)
    pyferret.stats.assignpmf(result, resbdf, distrib, inputs[0], inpbdfs[0])


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # Binomial distribution along the Y axis
    pfname = "binom"
    pfparams = numpy.array([25, 0.15], dtype=numpy.float32)
    distf = scipy.stats.binom(25, 0.15)
    xvals = numpy.arange(0.0, 25.1, 1.0)
    pmfvals = distf.pmf(xvals)
    abscissa = numpy.empty((1, 26, 1, 1), dtype=numpy.float32, order='F')
    expected = numpy.empty((1, 26, 1, 1), dtype=numpy.float32, order='F')
    for j in xrange(26):
        if (j % 7) == 2:
            abscissa[0, j, 0, 0] = -1.0
            expected[0, j, 0, 0] = -2.0
        else:
            abscissa[0, j, 0, 0] = xvals[j]
            expected[0, j, 0, 0] = pmfvals[j]
    inpbdfs = numpy.array([-1.0, 0.0, 0.0], dtype=numpy.float32)

    result = -888.0 * numpy.ones((1, 26, 1, 1), dtype=numpy.float32, order='F')
    resbdf = numpy.array([-2.0], dtype=numpy.float32)

    ferret_compute(0, result, resbdf, (abscissa, pfname, pfparams), inpbdfs)

    print "Result (flattened) = %s" % str(result.reshape(-1))
    if not numpy.allclose(result, expected):
        print "Expected (flattened) = %s" % str(expected.reshape(-1))
        raise ValueError("Unexpected result")

    # All successful
    print "Success"

