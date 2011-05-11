"""
Returns the array of probability density function values for a given
probability distribution and set of abscissa values.
"""
import sys
import numpy
import scipy.stats
import pyferret
import pyferret.stats


def ferret_init(id):
    """
    Initialization for the statspdf python-backed ferret external function
    """
    retdict = { "numargs": 3,
                "descript": "Returns probability density function values for a specified distribution",
                "axes": (pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS),
                "argnames": ("Abscissae", "PDName", "PDParams"),
                "argdescripts": ("Points at which to calculate the probability density function values",
                                 "Probability distribution name",
                                 "Parameters for this probability distribution"),
                "argtypes": (pyferret.FLOAT_ARG, pyferret.STRING_ARG, pyferret.FLOAT_ARG),
                "influences": ((True,  True,  True,  True),
                               (False, False, False, False),
                               (False, False, False, False)),
              }
    return retdict


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Assigns result with the probability density function values for the
    probability distribution indicated by inputs[1] using the parameters
    given in inputs[2] at the abscissa values given by inputs[0].
    """
    distribname = inputs[1]
    distribparams = inputs[2].reshape(-1)
    distrib = pyferret.stats.getdistrib(distribname, distribparams)
    pyferret.stats.assignpdf(result, resbdf, distrib, inputs[0], inpbdfs[0])


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # Normal distribution along the Y axis
    pfname = "norm"
    pfparams = numpy.array([5.0, 2.0], dtype=numpy.float32)
    distf = scipy.stats.norm(5.0, 2.0)
    xvals = numpy.arange(0.0, 10.1, 0.5)
    pdfvals = distf.pdf(xvals)
    abscissa = numpy.empty((1, 21, 1, 1), dtype=numpy.float32, order='F')
    expected = numpy.empty((1, 21, 1, 1), dtype=numpy.float32, order='F')
    for j in xrange(21):
        if (j % 7) == 0:
            abscissa[0, j, 0, 0] = -1.0
            expected[0, j, 0, 0] = -2.0
        else:
            abscissa[0, j, 0, 0] = xvals[j]
            expected[0, j, 0, 0] = pdfvals[j]
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

