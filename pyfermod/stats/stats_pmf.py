"""
Returns the array of probability density function values for a
discrete probability distribution and set of abscissa values.
"""
import numpy
import scipy.stats
import pyferret
import pyferret.stats


def ferret_init(id):
    """
    Initialization for the stats_pmf python-backed ferret external function
    """
    retdict = { "numargs": 3,
                "descript": "Returns probability mass function values for a discrete probability distribution",
                "axes": (pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS),
                "argnames": ("PTS", "PDNAME", "PDPARAMS"),
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
    For undefined abscissa values, the result value will be undefined.
    """
    distribname = inputs[1]
    distribparams = inputs[2].reshape(-1)
    distrib = pyferret.stats.getdistrib(distribname, distribparams)
    badmask = ( numpy.fabs(inputs[0] - inpbdfs[0]) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(inputs[0]))
    goodmask = numpy.logical_not(badmask)
    result[badmask] = resbdf
    # array[goodmask] is a flattened array
    result[goodmask] = distrib.pmf(inputs[0][goodmask])


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # make sure ferret_init does not have problems
    info = ferret_init(0)

    # Binomial distribution along the Y axis
    numtrials = 25
    prob = 0.15
    dimen = 26
    distf = scipy.stats.binom(numtrials, prob)
    xvals = numpy.arange(0, dimen)
    pmfvals = distf.pmf(xvals)

    pfname = "binom"
    pfparams = numpy.array([numtrials, prob], dtype=numpy.float32)
    inpbdfs = numpy.array([-1.0, 0.0, 0.0], dtype=numpy.float32)
    resbdf = numpy.array([-2.0], dtype=numpy.float32)
    abscissa = numpy.empty((1, dimen, 1, 1), dtype=numpy.float32, order='F')
    expected = numpy.empty((1, dimen, 1, 1), dtype=numpy.float32, order='F')
    for j in xrange(dimen):
        if (j % 7) == 2:
            abscissa[0, j, 0, 0] = inpbdfs[0]
            expected[0, j, 0, 0] = resbdf[0]
        else:
            abscissa[0, j, 0, 0] = xvals[j]
            expected[0, j, 0, 0] = pmfvals[j]
    result = -888.0 * numpy.ones((1, dimen, 1, 1), dtype=numpy.float32, order='F')
    ferret_compute(0, result, resbdf, (abscissa, pfname, pfparams), inpbdfs)
    if not numpy.allclose(result, expected):
        print "Result (flattened) = %s" % str(result.reshape(-1))
        print "Expected (flattened) = %s" % str(expected.reshape(-1))
        raise ValueError("Unexpected result")

    # All successful
    print "Success"

