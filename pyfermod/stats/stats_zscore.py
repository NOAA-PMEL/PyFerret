"""
Returns the array of standard scores for an array of data.  The
standard score are for the standard distribution centered of the
mean value of the data with the same variance as the data.
"""
import math
import numpy
import pyferret
import scipy.stats


def ferret_init(id):
    """
    Initialization for the stats_zscore PyEF
    """
    retdict = { "numargs": 1,
                "descript": "Returns standard scores for data values relative to" \
                            "a normal distribution with same mean and variance as the data",
                "axes": ( pyferret.AXIS_IMPLIED_BY_ARGS,
                          pyferret.AXIS_IMPLIED_BY_ARGS,
                          pyferret.AXIS_IMPLIED_BY_ARGS,
                          pyferret.AXIS_IMPLIED_BY_ARGS, ),
                "argnames": ( "VALUES", ),
                "argdescripts": ( "Array of data values", ),
                "argtypes": ( pyferret.FLOAT_ARG, ),
                "influences": ( (True,  True,  True,  True), ),
              }
    return retdict


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Assigns result standard scores of data values given in inputs[0]
    relative to a normal distribution with the same mean and variance
    as the data.  For undefined data values, the result value will
    be undefined.
    """
    badmask = ( numpy.fabs(inputs[0] - inpbdfs[0]) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(inputs[0]))
    goodmask = numpy.logical_not(badmask)
    result[badmask] = resbdf
    # convert to 64-bit for precision in calculating the mean and variance
    sample = numpy.array(inputs[0][goodmask], dtype=numpy.float64)
    # array[goodmask] is a flattened array
    result[goodmask] = scipy.stats.zscore(sample)


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # make sure ferret_init does not have problems
    info = ferret_init(0)

    # Get a random sample and the expected standard scores
    ydim = 83
    zdim = 17
    samplesize = 1300
    sample = scipy.stats.norm(5.0, 2.0).rvs(samplesize)
    zscores = (sample - sample.mean()) / math.sqrt(sample.var(0))

    # setup for the call to ferret_compute
    inpbdfs = numpy.array([-9999.0], dtype=numpy.float32)
    resbdf = numpy.array([-8888.0], dtype=numpy.float32)
    input = numpy.empty((1, ydim, zdim, 1), dtype=numpy.float32, order='F')
    expected = numpy.empty((1, ydim, zdim, 1), dtype=numpy.float32, order='F')
    sindex = 0
    iindex = 0
    for j in xrange(ydim):
        for k in xrange(zdim):
            if ((iindex % 13) == 3) or (sindex >= samplesize):
                input[0, j, k, 0] = inpbdfs[0]
                expected[0, j, k, 0] = resbdf
            else:
                input[0, j, k, 0] = sample[sindex]
                expected[0, j, k, 0] = zscores[sindex]
                sindex += 1
            iindex += 1
    if sindex != samplesize:
        raise ValueError("Unexpected final sindex of %d (ydim,zdim too small)" % sindex)
    result = -7777.0 * numpy.ones((1, ydim, zdim, 1), dtype=numpy.float32, order='F')

    # call ferret_compute and check the results
    ferret_compute(0, result, resbdf, (input, ), inpbdfs)
    if not numpy.allclose(result, expected, rtol=2.0E-7, atol=2.0E-7):
        print "expected (flattened) =\n%s" % str(expected.reshape(-1, order='F'))
        print "result (flattened) =\n%s" % str(result.reshape(-1, order='F'))
        raise ValueError("Unexpected result")

    # All successful
    print "Success"

