"""
Returns interpolated percentiles through a sample of scores (values)
"""
import numpy
import pyferret
import scipy.stats


def ferret_init(id):
    """
    Initialization for the stats_percentilesofscores.py Ferret PyEF
    """
    retdict = { "numargs": 2,
                "descript": "Returns interpolated percentiles (0-100) through a sample for given scores (values)",
                "axes": ( pyferret.AXIS_IMPLIED_BY_ARGS,
                          pyferret.AXIS_IMPLIED_BY_ARGS,
                          pyferret.AXIS_IMPLIED_BY_ARGS,
                          pyferret.AXIS_IMPLIED_BY_ARGS, ),
                "argnames": ( "SAMPLE", "SCORES", ),
                "argdescripts": ( "Sample of scores (values)",
                                  "Scores (values) to find percentiles through sample", ),
                "argtypes": ( pyferret.FLOAT_ARRAY, pyferret.FLOAT_ARRAY, ),
                "influences": ( ( False, False, False, False, ),
                                ( True,  True,  True,  True, ), ),
              }
    return retdict


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Assigns result with interpolated percentiles through a sample
    that correspond to given scores.  The sample scores are given
    in inputs[0], and the scores to find percentiles of are given
    in inputs[0].  Undefined values in inputs[0] are eliminated
    before using them in scipy.stats.percentileofscore.  Undefined
    values in inputs[1] return corresponding undefined values in
    result.
    """
    # make sure result has the expected dimensions
    if result.shape != inputs[1].shape:
        raise ValueError("Unexpected error; SCORES dimen: %s; result dimen: %s" % \
                         (str(inputs[1].shape), str(result.shape)))
    # get the clean sample data as a flattened array
    badmask = ( numpy.fabs(inputs[0] - inpbdfs[0]) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(inputs[0]))
    goodmask = numpy.logical_not(badmask)
    # note that these are still float32 values
    values = inputs[0][goodmask]
    # get the mask for the good scores
    badmask = ( numpy.fabs(inputs[1] - inpbdfs[1]) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(inputs[1]))
    goodmask = numpy.logical_not(badmask)
    # percentileofscore doesn't take an array for the scores
    # so do them one at a time
    prcnts = [ ]
    for score in inputs[1][goodmask]:
        prcnts.append(scipy.stats.percentileofscore(values, score))
    result[goodmask] = prcnts
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
    inpbdfs = numpy.array([-1.0, -2.0], dtype=numpy.float32)
    resbdf = numpy.array([-3.0], dtype=numpy.float32)
    sample = numpy.empty((1, ydim, zdim, 1), dtype=numpy.float32, order='F')
    # valid sample values are [1:100:1] + offset
    pval = 1
    index = 0
    for j in xrange(ydim):
        for k in xrange(zdim):
            if ((index % 7) == 3) or (pval > 100):
                sample[0, j, k, 0] = inpbdfs[0]
            else:
                sample[0, j, k, 0] = pval + offset
                pval += 1
            index += 1
    if pval != 101:
        raise ValueError("Unexpected final pval of %d (ydim,zdim too small)" % pval)
    scores = numpy.empty((1, 1, zdim, 1), dtype=numpy.float32, order='F')
    expected = numpy.empty((1, 1, zdim, 1), dtype=numpy.float32, order='F')
    scores[:,:,:,:] = inpbdfs[1]
    expected[:,:,:,:] = resbdf
    for k in ( 1, 2, 3, 5, 6, 7, 9 ):
        scores[0, 0, k, 0] = 10.0 * k + offset
        expected[0, 0, k, 0] = 10.0 * k
    result = -888.0 * numpy.ones((1, 1, zdim, 1), dtype=numpy.float32, order='F')
    ferret_compute(0, result, resbdf, (sample, scores), inpbdfs)
    if not numpy.allclose(result, expected):
        print "Expected (flattened) = %s" % str(expected.reshape(-1))
        print "Result (flattened) = %s" % str(result.reshape(-1))
        raise ValueError("Unexpected result")

    # All successful
    print "Success"

