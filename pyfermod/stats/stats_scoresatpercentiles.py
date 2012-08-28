"""
Returns interpolated scores (values) at percentiles through a sample
"""
import numpy
import pyferret
import scipy.stats


def ferret_init(id):
    """
    Initialization for the stats_scoresatpercentiles.py Ferret PyEF
    """
    retdict = { "numargs": 2,
                "descript": "Returns interpolated scores (values) that are given percentiles through a sample",
                "axes": ( pyferret.AXIS_IMPLIED_BY_ARGS,
                          pyferret.AXIS_IMPLIED_BY_ARGS,
                          pyferret.AXIS_IMPLIED_BY_ARGS,
                          pyferret.AXIS_IMPLIED_BY_ARGS,
                          pyferret.AXIS_IMPLIED_BY_ARGS,
                          pyferret.AXIS_IMPLIED_BY_ARGS, ),
                "argnames": ( "SAMPLE", "PERCENTILES", ),
                "argdescripts": ( "Sample of scores (values)",
                                  "Percentiles (0-100) through sample to find scores (values) of", ),
                "argtypes": ( pyferret.FLOAT_ARRAY, pyferret.FLOAT_ARRAY, ),
                "influences": ( ( False, False, False, False, False, False, ),
                                ( True,  True,  True,  True,  True,  True, ), ),
              }
    return retdict


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Assigns result with interpolated scores that are given percentiles
    through a sample.  The sample scores are given in inputs[0], and the
    percentiles are given in inputs[1].  Undefined values in inputs[0]
    are eliminated before using it in scipy.stats.scoreatpercentile.
    Undefined values in inputs[1] return corresponding undefined values
    in result.
    """
    # make sure result has the expected dimensions
    if result.shape != inputs[1].shape:
        raise ValueError("Unexpected error; PERCENTILE dimen: %s; result dimen: %s" % \
                         (str(inputs[1].shape), str(result.shape)))
    # get the clean sample data as a flattened array
    badmask = ( numpy.fabs(inputs[0] - inpbdfs[0]) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(inputs[0]))
    goodmask = numpy.logical_not(badmask)
    values = inputs[0][goodmask]
    # get the mask for the good percentiles
    badmask = ( numpy.fabs(inputs[1] - inpbdfs[1]) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(inputs[1]))
    goodmask = numpy.logical_not(badmask)
    # scoreatpercentile doesn't take an array for the percentiles
    # so do them one at a time
    scores = [ ]
    for prcnt in inputs[1][goodmask]:
        scores.append(scipy.stats.scoreatpercentile(values, prcnt))
    result[goodmask] = scores
    result[badmask] = resbdf


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # make sure ferret_init does not have problems
    info = ferret_init(0)

    ydim = 10
    zdim = 12
    offset = 32.5
    inpbdfs = numpy.array([-1.0, -2.0], dtype=numpy.float64)
    resbdf = numpy.array([-3.0], dtype=numpy.float64)
    sample = numpy.empty((1, ydim, zdim, 1, 1, 1), dtype=numpy.float64, order='F')
    # valid sample values are [0:100:1] + offset
    pval = 0
    index = 0
    for j in xrange(ydim):
        for k in xrange(zdim):
            if ((index % 7) == 3) or (pval > 100):
                sample[0, j, k, 0, 0, 0] = inpbdfs[0]
            else:
                sample[0, j, k, 0, 0, 0] = pval + offset
                pval += 1
            index += 1
    if pval != 101:
        raise ValueError("Unexpected final pval of %d (ydim,zdim too small)" % pval)
    prcnts = numpy.empty((1, 1, zdim, 1, 1, 1), dtype=numpy.float64, order='F')
    expected = numpy.empty((1, 1, zdim, 1, 1, 1), dtype=numpy.float64, order='F')
    prcnts[:,:,:,:,:,:] = inpbdfs[1]
    expected[:,:,:,:,:,:] = resbdf
    for k in ( 1, 2, 3, 5, 6, 7, 9 ):
        prcnts[0, 0, k, 0, 0, 0] = 10.0 * k
        expected[0, 0, k, 0, 0, 0] = 10.0 * k + offset
    result = -888.0 * numpy.ones((1, 1, zdim, 1, 1, 1), dtype=numpy.float64, order='F')
    ferret_compute(0, result, resbdf, (sample, prcnts), inpbdfs)
    if not numpy.allclose(result, expected):
        print "Expected (flattened) = %s" % str(expected.reshape(-1))
        print "Result (flattened) = %s" % str(result.reshape(-1))
        raise ValueError("Unexpected result")

    # All successful
    print "Success"

