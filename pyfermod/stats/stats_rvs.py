"""
Returns the array of random variates for a probability distribution
assigned to positions corresponding to defined values in an input array.
"""
import numpy
import scipy.stats
import pyferret
import pyferret.stats


def ferret_init(id):
    """
    Initialization for the stats_rvs python-backed ferret external function
    """
    retdict = { "numargs": 3,
                "descript": "Returns random variates for a probability distribution",
                "axes": (pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS),
                "argnames": ("TEMPLATE", "PDNAME", "PDPARAMS"),
                "argdescripts": ("Template array for the array of random variates to be returned",
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
    Assigns result with random variates of the probability distribution
    indicated by inputs[1] (a string) using the parameters given in
    inputs[2].  Random variates will be assigned to positions corresponding
    to defined positions in inputs[0].  For positions where the inputs[0]
    value is undefined, the result value will be undefined.
    """
    distribname = inputs[1]
    distribparams = inputs[2].reshape(-1)
    distrib = pyferret.stats.getdistrib(distribname, distribparams)
    badmask = ( numpy.fabs(inputs[0] - inpbdfs[0]) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(inputs[0]))
    goodmask = numpy.logical_not(badmask)
    result[badmask] = resbdf
    # result[goodmask] is a flattened array
    result[goodmask] = distrib.rvs(len(result[goodmask]))


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # Get a large sample from a normal distribution with small variance
    # so the mean and stdev will be very close
    mu = 5.0
    sigma = 0.5
    # probability of randomly choosing -65536.0 is essentially zero
    undefval = -65536.0
    xdim = 19
    ydim = 21
    zdim = 17
    tdim = 23

    pfname = "norm"
    pfparams = numpy.array([mu, sigma], dtype=numpy.float32)
    inpbdfs = numpy.array([-1.0, 0.0, 0.0], dtype=numpy.float32)
    resbdf = numpy.array([undefval], dtype=numpy.float32)
    # template initialized to all zero != impbdfs[0]
    template = numpy.zeros((xdim, ydim, zdim, tdim), dtype=numpy.float32, order='F')
    expectedgood = numpy.empty((xdim, ydim, zdim, tdim), dtype=bool, order='F')
    index = 0
    for i in xrange(xdim):
        for j in xrange(ydim):
            for k in xrange(zdim):
                for l in xrange(tdim):
                    if (index % 53) == 1:
                        template[i, j, k, l] = inpbdfs[0]
                        expectedgood[i, j, k, l] = False
                    else:
                        expectedgood[i, j, k, l] = True
                    index += 1
    result = -8888.0 * numpy.ones((xdim, ydim, zdim, tdim), dtype=numpy.float32, order='F')
    ferret_compute(0, result, resbdf, (template, pfname, pfparams), inpbdfs)
    resultgood = ( result != resbdf )
    if numpy.any( resultgood !=  expectedgood ):
        raise ValueError("Assigned random variates does not match template")
    mean = numpy.mean(result[resultgood])
    if abs(mean - 5.0) > 5.0E-3:
        raise ValueError("Mean of random sample: expected: 5.0; found: %f" % mean)
    stdev = numpy.std(result[resultgood])
    if abs(stdev - 0.5) > 5.0E-3:
        raise ValueError("Standard deviation of random sample: expected: 0.5; found: %f" % stdev)

    # All successful
    print "Success"

