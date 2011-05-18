"""
Returns the array of cumulative distribution function values
for a probability distribution and set of abscissa values.
"""
import sys
import numpy
import scipy.stats
import pyferret
import pyferret.stats


def ferret_init(id):
    """
    Initialization for the stats_cdf python-backed ferret external function
    """
    retdict = { "numargs": 3,
                "descript": "Returns cumulative distribution function values for a probability distribution",
                "axes": (pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS),
                "argnames": ("ABSCISSAE", "PDNAME", "PDPARAMS"),
                "argdescripts": ("Points at which to calculate the cumulative distribution function values",
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
    Assigns result with the cumulative distribution function values for
    the probability distribution indicated by inputs[1] (a string) using
    the parameters given in inputs[2] at the abscissa values given by
    inputs[0].
    """
    distribname = inputs[1]
    distribparams = inputs[2].reshape(-1)
    distrib = pyferret.stats.getdistrib(distribname, distribparams)
    pyferret.stats.assigncdf(result, resbdf, distrib, inputs[0], inpbdfs[0])


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # Normal distribution along the Y axis
    pfname = "norm"
    pfparams = numpy.array([5.0, 2.0], dtype=numpy.float32)
    distf = scipy.stats.norm(5.0, 2.0)
    xvals = numpy.arange(0.0, 10.1, 0.5)
    cdfvals = distf.cdf(xvals)
    abscissa = numpy.empty((1, 21, 1, 1), dtype=numpy.float32, order='F')
    expected = numpy.empty((1, 21, 1, 1), dtype=numpy.float32, order='F')
    for j in xrange(21):
        if (j % 7) == 0:
            abscissa[0, j, 0, 0] = -1.0
            expected[0, j, 0, 0] = -2.0
        else:
            abscissa[0, j, 0, 0] = xvals[j]
            expected[0, j, 0, 0] = cdfvals[j]
    inpbdfs = numpy.array([-1.0, 0.0, 0.0], dtype=numpy.float32)

    result = -888.0 * numpy.ones((1, 21, 1, 1), dtype=numpy.float32, order='F')
    resbdf = numpy.array([-2.0], dtype=numpy.float32)

    ferret_compute(0, result, resbdf, (abscissa, pfname, pfparams), inpbdfs)

    if not numpy.allclose(result, expected):
        print "Expected (flattened) = %s" % str(expected.reshape(-1))
        print "Result (flattened) = %s" % str(result.reshape(-1))
        raise ValueError("Unexpected result")

    # All successful
    print "Success"

