"""
Performs a chi-square test that a sample with the observed
counts of categorical data comes from a population with the
given expected counts or relative frequencies of that data.
"""
import numpy
import pyferret
import scipy.stats


def ferret_init(id):
    """
    Initialization for the stats_chisquare Ferret PyEF
    """
    retdict = { "numargs": 3,
                "descript": "Returns chi-square test stat. and prob. (and num. good categories, N) " \
                            "that sample counts of cat. data matches pop. expected counts. ",
                "axes": ( pyferret.AXIS_CUSTOM,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST, ),
                "argnames": ( "SAMPLE_CNTS", "EXPECT_CNTS", "DELTA_DEGFREE", ),
                "argdescripts": ( "Sample counts of categorical data",
                                  "Expected counts or relative frequencies (will be adjusted)",
                                  "Difference from standard (N-1) degrees of freedom (num. computed parameters)", ),
                "argtypes": ( pyferret.FLOAT_ARG, pyferret.FLOAT_ARG, pyferret.FLOAT_ARG, ),
                "influences": ( (False, False, False, False),
                                (False, False, False, False),
                                (False, False, False, False), ),
              }
    return retdict


def ferret_custom_axes(id):
    """
    Define custom axis of the stats_chisquare Ferret PyEF
    """
    return ( ( 1, 3, 1, "X2,P,N", False ), None, None, None, )


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Performs a chi-square test that a sample with the observed counts
    of categorical data, given in inputs[0], comes from a population
    with the expected counts or relative frequencies given in inputs[1].
    The difference from the standard (n-1) degrees of freedom (eg, the
    number of population parameters estimated from the sample) are given
    in inputs[2].  The test statistic value and probability are returned
    in result.  The counts arrays must either have the same shape or both
    have a single defined non-sigular axis of the same size.  Categories
    that contain undefined counts are elimated before performing the
    test.  Population values that are used in the test are adjust so
    their sum equals the sum of the sample counts used in the test.
    """
    if inputs[0].shape != inputs[1].shape :
        errmsg = "SAMPLE_CNTS and EXPECT_CNTS must either have identical dimensions " \
                 "or both have only one defined non-singular axis of the same length"
        lensam = 1
        for k in xrange(4):
            if inputs[0].shape[k] > 1:
                if lensam != 1:
                    raise ValueError(errmsg)
                lensam = inputs[0].shape[k]
        lenpop = 1
        for k in xrange(4):
            if inputs[1].shape[k] > 1:
                if lenpop != 1:
                    raise ValueError(errmsg)
                lenpop = inputs[1].shape[k]
        if lensam != lenpop:
            raise ValueError(errmsg)
    samcnts = inputs[0].reshape(-1)
    popcnts = inputs[1].reshape(-1)
    badsam = ( numpy.fabs(samcnts - inpbdfs[0]) < 1.0E-5 )
    badsam = numpy.logical_or(badsam, numpy.isnan(samcnts))
    goodsam = numpy.logical_not(badsam)
    badpop = ( numpy.fabs(popcnts - inpbdfs[1]) < 1.0E-5 )
    badpop = numpy.logical_or(badpop, numpy.isnan(popcnts))
    goodpop = numpy.logical_not(badpop)
    goodmask = numpy.logical_and(goodsam, goodpop)
    # must use double precision arrays for accuracy
    samcnts = numpy.array(samcnts[goodmask], dtype=numpy.float64)
    numgood = len(samcnts)
    if numgood < 2:
        raise ValueError("Not enough defined counts in common in SAMPLE_CNTS and EXPECT_CNTS")
    popcnts = numpy.array(popcnts[goodmask], dtype=numpy.float64)
    # Adjust the expected counts so its sum matches the sum of the sample
    # counts;  thus expected counts can be proportions instead of counts
    # and removes issues about missing values.  Get the adjustment factor
    # from the means instead of the sums for accuracy.
    popcnts = popcnts * (samcnts.mean() / popcnts.mean())
    ddof = int(float(inputs[2][0, 0, 0, 0]) + 0.5)
    fitparams = scipy.stats.chisquare(samcnts, popcnts, ddof)
    result[:, :, :, :] = resbdf
    # chi-square test statistic
    result[0, 0, 0, 0] = fitparams[0]
    # probability
    result[1, 0, 0, 0] = fitparams[1]
    # number of good categories
    result[2, 0, 0, 0] = numgood


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # make sure ferret_init and ferret_custom_axes do not have problems
    info = ferret_init(0)
    info = ferret_custom_axes(0)

    # Get a sample histogram and expected frequencies
    ddof  = 3
    nbins = 90
    ssize = 100 * nbins
    distf = scipy.stats.weibull_min(2.0, 5.0)
    chival = 1000.0
    while chival > 100.0:
        sample = distf.rvs(ssize)
        bedges = distf.isf(numpy.linspace(0.95,0.05,nbins+1))
        (histgr, retedges) = numpy.histogram(sample, bins=bedges)
        histgr = numpy.array(histgr, dtype=numpy.float64)
        exphist = numpy.ones((nbins,), dtype=numpy.float64) * histgr.mean()
        chival = ((histgr - exphist)**2 / exphist).sum()
        print "created a sample with chival = %f" % chival
    prob = scipy.stats.chi2(nbins - 1 - ddof).sf(chival)
    expect = numpy.array([chival, prob, nbins], dtype=numpy.float32)
    print "sample histogram = \n%s" % str(histgr)
    print "expect histogram value for all bins = %f" % exphist[0]
    print "expect result = %s" % str(expect)

    # setup for the call to ferret_compute - one non-singular axis
    inpbdfs = numpy.array([-9999.0, -8888.0, -7777.0], dtype=numpy.float32)
    resbdf = numpy.array([-6666.0], dtype=numpy.float32)
    samhist = inpbdfs[0] * numpy.ones((1, 1, 2 * nbins, 1), dtype=numpy.float32, order='F')
    samhist[0, 0, ::2, 0] = histgr
    pophist = numpy.ones((1, 2 * nbins, 1, 1), dtype=numpy.float32, order='F')
    ddofarr = numpy.array([ddof], dtype=numpy.float32).reshape((1, 1, 1, 1), order='F')
    result = -5555.0 * numpy.ones((3, 1, 1, 1), dtype=numpy.float32, order='F')

    # call ferret_compute and check the result
    ferret_compute(0, result, resbdf, (samhist, pophist, ddofarr), inpbdfs)
    result = result.reshape(-1)
    print " found result = %s" % str(result)
    if not numpy.allclose(result, expect):
        raise ValueError("Unexpected result")

    # setup for the call to ferret_compute - multiple dimensions
    inpbdfs = numpy.array([-9999.0, -8888.0, -7777.0], dtype=numpy.float32)
    resbdf = numpy.array([-6666.0], dtype=numpy.float32)
    samhist = inpbdfs[0] * numpy.ones((1, 2, nbins, 1), dtype=numpy.float32, order='F')
    samhist[0, 0, ::2, 0] = histgr[0:nbins//2]
    samhist[0, 1, 1::2, 0] = histgr[nbins//2:]
    pophist = numpy.ones((1, 2, nbins, 1), dtype=numpy.float32, order='F')
    ddofarr = numpy.array([ddof], dtype=numpy.float32).reshape((1, 1, 1, 1), order='F')
    result = -5555.0 * numpy.ones((3, 1, 1, 1), dtype=numpy.float32, order='F')

    # call ferret_compute and check the result
    ferret_compute(0, result, resbdf, (samhist, pophist, ddofarr), inpbdfs)
    result = result.reshape(-1)
    print " found result = %s" % str(result)
    if not numpy.allclose(result, expect):
        raise ValueError("Unexpected result")

    # All successful
    print "Success"

