"""
Performs a two-sided T-test that the provided sample
comes from a population with the given mean(s).
"""
import numpy
import pyferret
import scipy.stats


def ferret_init(id):
    """
    Initialization for the stats_ttest1 PyEF
    """
    retdict = { "numargs": 2,
                "descript": "Returns [i=1] two-sided T-test stat. and [i=2] prob. " \
                            "for sample data coming from pop. with given mean(s).",
                "axes": ( pyferret.AXIS_CUSTOM,
                          pyferret.AXIS_CUSTOM,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST, ),
                "argnames": ( "SAMPLE", "POPMEANS", ),
                "argdescripts": ( "Sample data to compare",
                                  "Proposed population means (averages)", ),
                "argtypes": ( pyferret.FLOAT_ARRAY, pyferret.FLOAT_ARRAY, ),
                "influences": ( (False, False, False, False),
                                (False, False, False, False), ),
              }
    return retdict


def ferret_custom_axes(id):
    """
    Define custom axis of the stats_ttest1 Ferret PyEF
    """
    arglen = 1
    for axis in ( pyferret.X_AXIS, pyferret.Y_AXIS, pyferret.Z_AXIS, pyferret.T_AXIS ):
        axis_info = pyferret.get_axis_info(id, pyferret.ARG2, axis)
        num = axis_info.get("size", -1)
        if num > 0:
            arglen *= num
    # if all axes have undefined lengths, assume it is a single value
    return ( ( 1, 2, 1, "T,P", False ), (1, arglen, 1, "MEAN_INDEX", False), None, None, )


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Performs a two-sided T-test that the provided sampe comes from a
    population with the given mean values.  The sample is given in
    inputs[0] and the mean values are given in inputs[1].  The test
    statistic value and two-tailed probability are returned in result
    along the first axis for each mean along the second axis.
    Undefined data in inputs[0] are removed before performing the test.
    """
    # make sure result has the expected shape
    nummeans = len(inputs[1].reshape(-1))
    expected = (2, nummeans, 1, 1)
    if result.shape != expected:
        raise ValueError("Unexpected result dimensions; expect: %s, found: %s" % \
                         (str(expected), str(result.shapes)))
    # get the valid sample values as 64-bit floats
    badmask = ( numpy.fabs(inputs[0] - inpbdfs[0]) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(inputs[0]))
    goodmask = numpy.logical_not(badmask)
    values = numpy.array(inputs[0][goodmask], dtype=numpy.float64)
    # get the good mean values
    # need to flatten so the mask is one-dimensional
    means = inputs[1].reshape(-1)
    badmask = ( numpy.fabs(means - inpbdfs[1]) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(means))
    goodmask = numpy.logical_not(badmask)
    means = numpy.array(means[goodmask], dtype=numpy.float64)
    # perform the test and assign the results
    fitparams = scipy.stats.ttest_1samp(values, means)
    result[:, :, :, :] = resbdf
    # T-test statistics
    result[0, goodmask, 0, 0] = fitparams[0]
    # probabilities
    result[1, goodmask, 0, 0] = fitparams[1]


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # make sure ferret_init does not have problems
    info = ferret_init(0)

    # Get a random sample from a normal distribution
    ydim = 200
    zdim = 150
    mu = 5.0
    sigma = 2.0
    sample = scipy.stats.norm(mu, sigma).rvs(ydim * zdim)
    means = numpy.linspace(mu - sigma, mu + sigma, 5)

    # setup for the call to ferret_compute
    inpbdfs = numpy.array([-9999.0, -8888.0], dtype=numpy.float32)
    resbdf  = numpy.array([-7777.0], dtype=numpy.float32)
    samparr = numpy.empty((1, ydim, zdim, 1), dtype=numpy.float32, order='F')
    trimsamp = [ ]
    index = 0
    for j in xrange(ydim):
        for k in xrange(zdim):
            if (index % 71) == 3:
                samparr[0, j, k, 0] = inpbdfs[0]
            else:
                samparr[0, j, k, 0] = sample[index]
                trimsamp.append(sample[index])
            index += 1

    # computing each t-stat and p-val individually to check array arg return
    tvals = [ ]
    pvals = [ ]
    for m in means:
       (t, p) = scipy.stats.ttest_1samp(trimsamp, m)
       tvals.append(t)
       pvals.append(p)

    meanarr = inpbdfs[1] * numpy.ones((1, 2, 5, 1), dtype=numpy.float32, order='F')
    expect = resbdf * numpy.ones((2, 10, 1, 1), dtype=numpy.float32, order='F')
    meanarr[0, 0, 1, 0] = means[0]
    expect[:, 1, 0, 0] = [ tvals[0], pvals[0] ]
    meanarr[0, 0, 3, 0] = mu - 0.5 * sigma
    expect[:, 3, 0, 0] = [ tvals[1], pvals[1] ]
    meanarr[0, 1, 0, 0] = mu
    expect[:, 5, 0, 0] = [ tvals[2], pvals[2] ]
    meanarr[0, 1, 2, 0] = mu + 0.5 * sigma
    expect[:, 7, 0, 0] = [ tvals[3], pvals[3] ]
    meanarr[0, 1, 4, 0] = mu + sigma
    expect[:, 9, 0, 0] = [ tvals[4], pvals[4] ]
    result = -6666.0 * numpy.ones((2, 10, 1, 1), dtype=numpy.float32, order='F')

    # call ferret_compute and check the results
    ferret_compute(0, result, resbdf, (samparr, meanarr), inpbdfs)
    if not numpy.allclose(result, expect):
        print "result[:,:,0,0]:\n   %s" % str(result[:, :, 0, 0])
        print "expect[:,:,0,0]:\n   %s" % str(expect[:, :, 0, 0])
        raise ValueError("Unexpected result")

    # All successful
    print "Success"

