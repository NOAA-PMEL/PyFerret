"""
Performs a two-sided Kolmogorov-Smirnov test that two samples
come from the same continuous probability distribution.
"""
import numpy
import pyferret
import scipy.stats


def ferret_init(id):
    """
    Initialization for the stats_kstest2 PyEF
    """
    retdict = { "numargs": 2,
                "descript": "Returns two-sided Kolmogorov-Smirnov test stat. and prob. " \
                            "that two samples comes from the same prob. distrib.",
                "axes": ( pyferret.AXIS_CUSTOM,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST, ),
                "argnames": ( "SAMPLEA", "SAMPLEB", ),
                "argdescripts": ( "First sample data array",
                                  "Second sample data array", ),
                "argtypes": ( pyferret.FLOAT_ARRAY, pyferret.FLOAT_ARRAY, ),
                "influences": ( (False, False, False, False),
                                (False, False, False, False), ),
              }
    return retdict


def ferret_custom_axes(id):
    """
    Define custom axis of the stats_kstest2 Ferret PyEF
    """
    return ( ( 1, 2, 1, "KS,P", False ), None, None, None, )


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Performs a two-sided Kolmogorov-Smirnov test that two samples come
    from the same continuous probability distribution.  The samples are
    given in inputs[0] and inputs[1].  The test statistic value and
    two-tailed probability are returned in result.  Undefined data given
    in each sample are removed (independently from each other) before
    performing the test.  Note that the samples do not need to be the
    same size; thus there are no restrictions on the relative dimensions
    of sample arrays.
    """
    badmask = ( numpy.fabs(inputs[0] - inpbdfs[0]) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(inputs[0]))
    goodmask = numpy.logical_not(badmask)
    sampa = inputs[0][goodmask]
    badmask = ( numpy.fabs(inputs[1] - inpbdfs[1]) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(inputs[1]))
    goodmask = numpy.logical_not(badmask)
    sampb = inputs[1][goodmask]
    fitparams = scipy.stats.ks_2samp(sampa, sampb)
    result[:, :, :, :] = resbdf
    # Kolmogorov-Smirnov test statistic
    result[0, 0, 0, 0] = fitparams[0]
    # probability
    result[1, 0, 0, 0] = fitparams[1]


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # make sure ferret_init and ferret_custom_axes do not have problems
    info = ferret_init(0)
    info = ferret_custom_axes(0)

    # Get two random samples from the same distribution
    ydim = 250
    zdim = 150
    distf = scipy.stats.weibull_min(2.0, 5.0)
    sampa = distf.rvs(ydim * zdim)
    sampb = distf.rvs(ydim * zdim)

    # Get a distribution from a different distribution
    sampu = scipy.stats.uniform(loc=7.0, scale=5.0).rvs(ydim * zdim)

    # setup for the call to ferret_compute
    inpbdfs = numpy.array([-9999.0, -8888.0], dtype=numpy.float32)
    resbdf = numpy.array([-7777.0], dtype=numpy.float32)
    arraya = numpy.empty((1, ydim, zdim, 1), dtype=numpy.float32, order='F')
    arrayb = numpy.empty((ydim, 1, 1, zdim), dtype=numpy.float32, order='F')
    arrayu = numpy.empty((ydim, 1, 1, zdim), dtype=numpy.float32, order='F')
    index = 0
    for j in xrange(ydim):
        for k in xrange(zdim):
            if (index % 23) == 3:
                arraya[0, j, k, 0] = inpbdfs[0]
            else:
                arraya[0, j, k, 0] = sampa[index]
            if (index % 53) == 3:
                arrayb[j, 0, 0, k] = inpbdfs[1]
                arrayu[j, 0, 0, k] = inpbdfs[1]
            else:
                arrayb[j, 0, 0, k] = sampb[index]
                arrayu[j, 0, 0, k] = sampu[index]
            index += 1
    resultb = -6666.0 * numpy.ones((2, 1, 1, 1), dtype=numpy.float32, order='F')
    resultu = -6666.0 * numpy.ones((2, 1, 1, 1), dtype=numpy.float32, order='F')

    # call ferret_compute with the samples from the same distribution and check the results
    ferret_compute(0, resultb, resbdf, (arraya, arrayb), inpbdfs)
    resultb = resultb.reshape(-1)
    print "from same dist result: %s" % str(resultb)
    if (resultb[0] < 0.00) or (resultb[0] > 0.01) or \
       (resultb[1] < 0.10) or (resultb[1] > 1.00):
        raise ValueError("Unexpected result")

    # call ferret_compute with data from different distributions and check the results
    ferret_compute(0, resultu, resbdf, (sampa, sampu), inpbdfs)
    resultu = resultu.reshape(-1)
    print "from diff dist result: %s" % str(resultu)
    if (resultu[0] < 0.98) or (resultu[0] > 1.00) or \
       (resultu[1] < 0.00) or (resultu[1] > 0.01):
        raise ValueError("Unexpected result")

    # All successful
    print "Success"

