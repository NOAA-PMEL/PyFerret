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
                "argnames": ("PDName", "PDParams", "Abscissae"),
                "argdescripts": ("Probability distribution name: 1-normal, 2-chisq",
                                 "Parameters for this probability distribution",
                                 "Points at which to calculate the probability density function values"),
		"influences": ((False, False, False, False),
		               (False, False, False, False),
		               (True,  True,  True,  True)),
              }
    return retdict


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Assigns result with the probability density function values for the
    probability distribution indicated by inputs[0] using the parameters
    given in inputs[1] at the abscissa values given by inputs[2].
    """
    distribname = (inputs[0].reshape(-1))[0]
    distribparams = inputs[1].reshape(-1)
    distrib = pyferret.stats.getdistrib(distribname, distribparams)
    badmask = ( numpy.fabs(inputs[2] - inpbdfs[2]) < 1.0E-5 )
    goodmask = numpy.logical_not(badmask)
    result[badmask] = resbdf
    # array[goodmask] is a 1-D array
    result[goodmask] = distrib.pdf(inputs[2][goodmask])


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    print "ferret_init(0) returned: \n%s" % str(ferret_init(0))

    # Normal distribution along the Y axis
    pfnum = numpy.array([1.0], dtype=numpy.float32)
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
    inpbdfs = numpy.array([0.0, 0.0, -1.0], dtype=numpy.float32)

    result = -888.0 * numpy.ones((1, 21, 1, 1), dtype=numpy.float32, order='F')
    resbdf = numpy.array([-2.0], dtype=numpy.float32)

    ferret_compute(0, result, resbdf, (pfnum, pfparams, abscissa), inpbdfs)

    print "Expected (flattened) = %s" % str(expected.reshape(-1))
    print "Result (flattened) = %s" % str(result.reshape(-1))

    if not numpy.allclose(result, expected):
        raise ValueError, "Unexpected result"

    # Chi-squared distribution along the Z axis
    pfnum = numpy.array([2.0], dtype=numpy.float32)
    pfparams = numpy.array([2.0], dtype=numpy.float32)
    distf = scipy.stats.chi2(2.0)
    xvals = numpy.arange(0.5, 10.6, 0.5)
    pdfvals = distf.pdf(xvals)
    abscissa = numpy.empty((1, 1, 21, 1), dtype=numpy.float32, order='F')
    expected = numpy.empty((1, 1, 21, 1), dtype=numpy.float32, order='F')
    for k in xrange(21):
        if (k % 7) == 1:
            abscissa[0, 0, k, 0] = -1.0
            expected[0, 0, k, 0] = -2.0
        else:
            abscissa[0, 0, k, 0] = xvals[k]
            expected[0, 0, k, 0] = pdfvals[k]
    inpbdfs = numpy.array([0.0, 0.0, -1.0], dtype=numpy.float32)

    result = -888.0 * numpy.ones((1, 1, 21, 1), dtype=numpy.float32, order='F')
    resbdf = numpy.array([-2.0], dtype=numpy.float32)

    ferret_compute(0, result, resbdf, (pfnum, pfparams, abscissa), inpbdfs)

    print "Expected (flattened) = %s" % str(expected.reshape(-1))
    print "Result (flattened) = %s" % str(result.reshape(-1))

    if not numpy.allclose(result, expected):
        raise ValueError, "Unexpected result"

    # All successful
    print "Success"

