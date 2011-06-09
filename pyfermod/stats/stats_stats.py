"""
Returns the (unweighted) mean, variance, skew, and kurtoses
of an array of values
"""
import math
import numpy
import pyferret
import scipy.stats


def ferret_init(id):
    """
    Initialization for the stats_stats.py Ferret PyEF
    """
    retdict = { "numargs": 1,
                "descript": "Returns the (unweighted) mean, variance, skew, and excess kurtosis of an array of values",
                "axes": ( pyferret.AXIS_CUSTOM,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST, ),
                "argnames": ( "VALUES", ),
                "argdescripts": ( "Array of values to find the statistical values of", ),
                "argtypes": ( pyferret.FLOAT_ARG, ),
                "influences": ( ( False, False, False, False, ), )
              }
    return retdict


def ferret_custom_axes(id):
    """
    Define custom axis of the stats_stats.py Ferret PyEF
    """
    return ( ( 1, 4, 1, "M,V,S,K", False ), None, None, None, )


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Assigns result with the (unweighted) mean, variance, skew, and
    excess kurtosis of the sample given in inputs[0].  Undefined
    values in inputs[0] are eliminated before using them in python
    methods.
    """
    if result.shape != (4, 1, 1, 1):
        raise ValueError("Unexpected error; result dimen: %s" % str(result.shape))
    # get the clean sample data as a flattened array
    badmask = ( numpy.fabs(inputs[0] - inpbdfs[0]) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(inputs[0]))
    goodmask = numpy.logical_not(badmask)
    # convert to 64-bit values for precision
    values = numpy.array(inputs[0][goodmask], dtype=numpy.float64)
    # Use the numpy/scipy methods which includes some guards
    result[0,0,0,0] = numpy.mean(values)
    result[1,0,0,0] = numpy.var(values)
    result[2,0,0,0] = scipy.stats.skew(values)
    result[3,0,0,0] = scipy.stats.kurtosis(values)


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # make sure ferret_init does not have problems
    info = ferret_init(0)
    info = ferret_custom_axes(0)

    # get a sample of non-normal data
    ydim = 83
    zdim = 17
    samplesize = 1300
    sample = scipy.stats.weibull_min(2.0, 5.0).rvs(samplesize)

    # use the traditional formulae (reasonable sample)
    mean = sample.mean()
    deltas = sample - mean
    vari = numpy.power(deltas, 2).mean()
    skew = numpy.power(deltas, 3).mean() / math.pow(vari, 1.5)
    kurt = (numpy.power(deltas, 4).mean() / math.pow(vari, 2)) - 3.0

    # setup for the call to ferret_compute
    inpbdfs = numpy.array([-9999.0], dtype=numpy.float32)
    resbdf = numpy.array([-8888.0], dtype=numpy.float32)
    input = numpy.empty((1, ydim, zdim, 1), dtype=numpy.float32, order='F')
    sindex = 0
    iindex = 0
    for j in xrange(ydim):
        for k in xrange(zdim):
            if ((iindex % 13) == 3) or (sindex >= samplesize):
                input[0, j, k, 0] = inpbdfs[0]
            else:
                input[0, j, k, 0] = sample[sindex]
                sindex += 1
            iindex += 1
    if sindex != samplesize:
        raise ValueError("Unexpected final sindex of %d (ydim,zdim too small)" % sindex)
    expected = numpy.array((mean, vari, skew, kurt), dtype=numpy.float32).reshape((4, 1, 1, 1), order='F')
    result = -7777.0 * numpy.ones((4, 1, 1, 1), dtype=numpy.float32, order='F')

    # call ferret_compute and check the results
    ferret_compute(0, result, resbdf, (input, ), inpbdfs)
    if not numpy.allclose(result, expected):
        raise ValueError("Unexpected result; expected: %s; found: %s" % \
                         (str(expected), str(result)))

    # All successful
    print "Success"

