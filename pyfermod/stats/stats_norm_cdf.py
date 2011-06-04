"""
Returns the array of cumulative density function values for the normal
probability distribution using given arrays of abscissa values, mu values,
and sigma value.
"""
import numpy
import pyferret
import pyferret.stats

DISTRIB_NAME = "Normal"
FUNC_NAME = "cdf"


def ferret_init(id):
    """
    Initialization for the stats_<distribname>_<funcname> Ferret PyEF
    """
    return pyferret.stats.getinitdict(DISTRIB_NAME, FUNC_NAME)


def ferret_custom_axes(id):
    """
    Custom axis definitions for the stats_<distribname>_<funcname> Ferret PyEF
    """
    return pyferret.stats.getcustomaxisvals(id, DISTRIB_NAME);


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Result array assignment for the stats_<distribname>_<funcname> Ferret PyEF
    """
    pyferret.stats.assignresultsarray(DISTRIB_NAME, FUNC_NAME, result,
                                      resbdf, inputs, inpbdfs)


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # make sure the calls to ferret_init does not cause problems
    dummy = ferret_init(0)
    # test ferret_compute
    xvals = numpy.arange(-2.0, 2.1, 1.0)
    muvals = numpy.arange(-1.5, 1.6, 1.0)
    # first sigma is invalid
    sigmavals = numpy.arange(-0.1, 1.0, 0.5)
    # bad/missing data flags
    inpbdfs = numpy.array([-9999.0, -8888.0, -7777.0], dtype=numpy.float32)
    resbdf = numpy.array([-6666.0], dtype=numpy.float32)
    # Compute the cdfs for xvals, muvals, sigmavals - first sigmaval is invalid
    cdfvals = numpy.empty((len(xvals), len(muvals), len(sigmavals)), dtype=numpy.float32, order='F')
    cdfvals[:,:,0] = resbdf
    for j in xrange(len(muvals)):
        for k in xrange(1, len(sigmavals)):
            distrib = pyferret.stats.getdistrib("norm", ( muvals[j], sigmavals[k], ))
            cdfvals[:,j,k] = distrib.cdf(xvals)
    # Create and assign the X array to be given to ferret_compute, as well as the expected results array
    abscissa = numpy.empty((len(xvals), 2, 1, 1), dtype=numpy.float32, order='F')
    expected = numpy.empty((2*len(xvals), len(muvals), len(sigmavals), 1), dtype=numpy.float32, order='F')
    for i in xrange(len(xvals)):
        if ( i % 2 ) == 1 :
            abscissa[i, 0, 0, 0] = inpbdfs[0]
            expected[i, :, :, 0] = resbdf
            abscissa[i, 1, 0, 0] = inpbdfs[0]
            expected[len(xvals) + i, :, :, 0] = resbdf
        else:
            abscissa[i, 0, 0, 0] = xvals[i//2]
            expected[i, :, :, 0] = cdfvals[i//2,:,:]
            abscissa[i, 1, 0, 0] = xvals[(len(xvals) + i)//2]
            expected[len(xvals) + i, :, :, 0] = cdfvals[(len(xvals) + i)//2,:,:]
    # Create the result array with garbage values
    result = -5555.0 * numpy.ones((2*len(xvals), len(muvals), len(sigmavals), 1), dtype=numpy.float32, order='F')
    # Run the calculation
    ferret_compute(0, result, resbdf, (abscissa, muvals, sigmavals), inpbdfs)
    # Compare results
    if not numpy.allclose(result, expected):
        print "Expected.reshape(-1, order='F') = %s" % str(expected.reshape(-1, order='F'))
        print "Result.reshape(-1, order='F') = %s " % str(result.reshape(-1, order='F'))
        raise ValueError("Unexpected result")
    # All successful
    print "Success"

