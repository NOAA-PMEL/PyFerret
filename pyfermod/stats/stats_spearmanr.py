"""
Returns Spearman's rank correlation coefficient between two samples of data.
"""
import math
import numpy
import pyferret
import scipy.stats


def ferret_init(id):
    """
    Initialization for the stats_spearmanr PyEF
    """
    retdict = { "numargs": 2,
                "descript": "Returns Spearman's rank correlation coeff, " \
                            "and num good points, between two samples of data",
                "axes": ( pyferret.AXIS_CUSTOM,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST, ),
                "argnames": ( "SAMPLEA", "SAMPLEB", ),
                "argdescripts": ( "First array of sample data",
                                  "Second array of sample data", ),
                "argtypes": ( pyferret.FLOAT_ARRAY, pyferret.FLOAT_ARRAY, ),
                "influences": ( (False, False, False, False),
                                (False, False, False, False), ),
              }
    return retdict


def ferret_custom_axes(id):
    """
    Define custom axis of the stats_spearmanr Ferret PyEF
    """
    return ( ( 1, 2, 1, "R,N", False ), None, None, None, )


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Assigns result with Spearman's rank correlation coefficient,
    and the number of good point, between the two samples of
    data given in inputs[0] and inputs[1].  Values compared are
    only from positions that are defined in both arrays.
    """
    if inputs[0].shape != inputs[1].shape :
        errmsg = "SAMPLEA and SAMPLEB must either have identical dimensions or "\
            "both have only one defined non-singular axis of the same length"
        lena = 1
        lenb = 1
        for k in xrange(4):
            if inputs[0].shape[k] > 1:
                if lena != 1:
                    raise ValueError(errmsg)
                lena = inputs[0].shape[k]
        for k in xrange(4):
            if inputs[1].shape[k] > 1:
                if lenb != 1:
                    raise ValueError(errmsg)
                lenb = inputs[1].shape[k]
        if lena != lenb:
            raise ValueError(errmsg)
    sampa = inputs[0].reshape(-1)
    sampb = inputs[1].reshape(-1)
    bada = ( numpy.fabs(sampa - inpbdfs[0]) < 1.0E-5 )
    bada = numpy.logical_or(bada, numpy.isnan(sampa))
    badb = ( numpy.fabs(sampb - inpbdfs[1]) < 1.0E-5 )
    badb = numpy.logical_or(badb, numpy.isnan(sampb))
    goodmask = numpy.logical_not(numpy.logical_or(bada, badb))
    # must use double precision arrays for accuracy
    valsa = numpy.array(sampa[goodmask], dtype=numpy.float64)
    numpts = len(valsa)
    if numpts < 2:
        raise ValueError("Not enough defined points in common in SAMPLEA and SAMPLEB")
    valsb = numpy.array(sampb[goodmask], dtype=numpy.float64)
    fitparams = scipy.stats.spearmanr(valsa, valsb)
    result[:] = resbdf
    # correlation coefficient
    result[0] = fitparams[0]
    # ignore the probability of uncorrelated
    # number of good pts
    result[1] = numpts


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # make sure ferret_init and ferret_custom_axes do not have problems
    info = ferret_init(0)
    info = ferret_custom_axes(0)

    # Get a random sample from a normal distribution
    ydim = 83
    zdim = 17
    samplesize = ydim * zdim
    sampa = scipy.stats.norm(5.0, 2.0).rvs(samplesize)

    # Create a correlated distribution
    sampc = -numpy.log(sampa)

    # Create an uncorrelated distribution and approx. Pearson Correlation Coeff.
    sampu = scipy.stats.norm(5.0, 2.0).rvs(samplesize)

    # setup for the call to ferret_compute
    inpbdfs = numpy.array([-9999.0, -8888.0], dtype=numpy.float32)
    resbdf = numpy.array([-7777.0], dtype=numpy.float32)
    inputa = numpy.empty((1, ydim, zdim, 1), dtype=numpy.float32, order='F')
    inputc = numpy.empty((1, ydim, zdim, 1), dtype=numpy.float32, order='F')
    inputu = numpy.empty((1, ydim, zdim, 1), dtype=numpy.float32, order='F')
    index = 0
    numgood = 0
    numpos = 0
    for j in xrange(ydim):
        for k in xrange(zdim):
            if (index % 23) == 3:
                inputa[0, j, k, 0] = inpbdfs[0]
            else:
                inputa[0, j, k, 0] = sampa[index]
            if (index % 31) == 3:
                inputc[0, j, k, 0] = inpbdfs[1]
                inputu[0, j, k, 0] = inpbdfs[1]
            else:
                inputc[0, j, k, 0] = sampc[index]
                inputu[0, j, k, 0] = sampu[index]
            if ((index % 23) != 3) and ((index % 31) != 3):
                numgood += 1
                if sampa[index] > 0.0:
                    numpos += 1
            index += 1
    resultc = -6666.0 * numpy.ones((2, 1, 1, 1), dtype=numpy.float32, order='F')
    expectc = numpy.empty((2, 1, 1, 1), dtype=numpy.float32, order='F')
    expectc[0,0,0,0] = -1.0
    expectc[1,0,0,0] = numpos
    resultu = -6666.0 * numpy.ones((2, 1, 1, 1), dtype=numpy.float32, order='F')
    expectu = numpy.empty((2, 1, 1, 1), dtype=numpy.float32, order='F')
    # rough expected correlation coefficient for uncorrelated
    expectu[0,0,0,0] = 0.0
    expectu[1,0,0,0] = numgood

    # call ferret_compute with correlated data and check the results
    ferret_compute(0, resultc, resbdf, (inputa, inputc), inpbdfs)
    if not numpy.allclose(resultc, expectc):
        raise ValueError("Unexpected result; expected: %s; found %s" % \
                         (str(expectc.reshape(-1)), str(resultc.reshape(-1))))

    # call ferret_compute with uncorrelated data and check the results
    ferret_compute(0, resultu, resbdf, (inputa, inputu), inpbdfs)
    if not numpy.allclose(resultu, expectu, atol=0.08):
        raise ValueError("Unexpected result; expected: %s; found %s" % \
                         (str(expectu.reshape(-1)), str(resultu.reshape(-1))))

    # All successful
    print "Success"

