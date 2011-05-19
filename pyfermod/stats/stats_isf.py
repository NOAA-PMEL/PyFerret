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
                "descript": "Returns inverse suvivial function values for a probability distribution",
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
    # Normal distribution along the Y axis
    pfname = "norm"
    pfparams = numpy.array([5.0, 2.0], dtype=numpy.float32)
    distf = scipy.stats.norm(5.0, 2.0)
    qvals = numpy.arange(0.05, 0.951, 0.05)
    isfvals = distf.isf(qvals)
    quantiles = numpy.empty((1, 19, 1, 1), dtype=numpy.float32, order='F')
    expected = numpy.empty((1, 19, 1, 1), dtype=numpy.float32, order='F')
    for j in xrange(19):
        if (j % 7) == 2:
            quantiles[0, j, 0, 0] = -1.0
            expected[0, j, 0, 0] = -2.0
        else:
            quantiles[0, j, 0, 0] = qvals[j]
            expected[0, j, 0, 0] = isfvals[j]
    inpbdfs = numpy.array([-1.0, 0.0, 0.0], dtype=numpy.float32)

    result = -888.0 * numpy.ones((1, 19, 1, 1), dtype=numpy.float32, order='F')
    resbdf = numpy.array([-2.0], dtype=numpy.float32)

    ferret_compute(0, result, resbdf, (quantiles, pfname, pfparams), inpbdfs)

    if not numpy.allclose(result, expected):
        print "Expected (flattened) = %s" % str(expected.reshape(-1))
        print "Result (flattened) = %s" % str(result.reshape(-1))
        raise ValueError("Unexpected result")

    # All successful
    print "Success"

