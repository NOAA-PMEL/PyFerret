"""
Returns the array of inverse survival function values for
a probability distribution and set of quantile values.
"""
import numpy
import scipy.stats
import pyferret
import pyferret.stats


def ferret_init(id):
    """
    Initialization for the stats_isf python-backed ferret external function
    """
    retdict = { "numargs": 3,
                "descript": "Returns inverse survival function values for a probability distribution",
                "axes": (pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS),
                "argnames": ("PROBS", "PDNAME", "PDPARAMS"),
                "argdescripts": ("Probabilities (0-1) at which to calculate the inverse survival function values",
                                 "Name of a probability distribution",
                                 "Parameters for this probability distribution"),
                "argtypes": (pyferret.FLOAT_ARG, pyferret.STRING_ARG, pyferret.FLOAT_ARG),
                "influences": ((True,  True,  True,  True),
                               (False, False, False, False),
                               (False, False, False, False)),
              }
    return retdict


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Assigns result with the inverse survival function values for the probability
    distribution indicated by inputs[1] (a string) using the parameters given in
    inputs[2] at the quantile values given by inputs[0].  For undefined quantile
    values, the result value will be undefined.
    """
    distribname = inputs[1]
    distribparams = inputs[2].reshape(-1)
    distrib = pyferret.stats.getdistrib(distribname, distribparams)
    badmask = ( numpy.fabs(inputs[0] - inpbdfs[0]) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(inputs[0]))
    goodmask = numpy.logical_not(badmask)
    result[badmask] = resbdf
    # array[goodmask] is a flattened array
    result[goodmask] = distrib.isf(inputs[0][goodmask])


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
    qvals = numpy.linspace(0.05, 0.95, dimen)
    isfvals = distf.isf(qvals)

    pfname = "norm"
    pfparams = numpy.array([mu, sigma], dtype=numpy.float32)
    inpbdfs = numpy.array([-1.0, 0.0, 0.0], dtype=numpy.float32)
    resbdf = numpy.array([-2.0], dtype=numpy.float32)
    quantile = numpy.empty((1, dimen, 1, 1), dtype=numpy.float32, order='F')
    expected = numpy.empty((1, dimen, 1, 1), dtype=numpy.float32, order='F')
    for j in xrange(dimen):
        if (j % 7) == 3:
            quantile[0, j, 0, 0] = inpbdfs[0]
            expected[0, j, 0, 0] = resbdf[0]
        else:
            quantile[0, j, 0, 0] = qvals[j]
            expected[0, j, 0, 0] = isfvals[j]
    result = -888.0 * numpy.ones((1, dimen, 1, 1), dtype=numpy.float32, order='F')
    ferret_compute(0, result, resbdf, (quantile, pfname, pfparams), inpbdfs)
    if not numpy.allclose(result, expected):
        print "Expected (flattened) = %s" % str(expected.reshape(-1))
        print "Result (flattened) = %s" % str(result.reshape(-1))
        raise ValueError("Unexpected result")

    # All successful
    print "Success"

