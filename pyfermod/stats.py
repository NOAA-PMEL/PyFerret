"""
Helper functions for pyferret stats external functions.
"""
import math
import numpy
import scipy.stats


def getdistrib(distribname, distribparams):
    """
    Creates and returns scipy.stats probability distribution object.

    Arguments:
       distribname - name of the distribution
       distribparams - tuple/list/array of input parameters

    Returns: 
       the scipy.stats distribution object named in distribname
       with the parameters given in distribparams

    Raises:
       ValueError if the distribution name is not recognized by
                  this routine or if the distribution parameters
                  are not the appropriate type
       IndexError if too few distribution parameters are given
    """
    lcdistname = str(distribname).lower()
    if (lcdistname == "norm") or (lcdistname == "normal"):
        mu = float(distribparams[0])
        sigma = float(distribparams[1])
        distrib = scipy.stats.norm(mu, sigma)
    elif (lcdistname == "chi2") or (lcdistname == "chi-square"):
        degfree = float(distribparams[0])
        distrib = scipy.stats.chi2(degfree)
    elif lcdistname == "poisson":
        lambdaflt = float(distribparams[0])
        distrib = scipy.stats.poisson(lambdaflt)
    else:
        raise ValueError("Unknown probability function %s" % str(distribname))
    return distrib


def assignpdf(result, resbdf, distrib, input, inpbdf):
    """
    Assigns the probability density function values of a distribution
    at specified positions.  At undefined positions, the results will
    be assigned as undefined.

    Arguments:
        result  - the numpy.ndarray to be assigned with the pdf values
        resbdf  - the undefined value for result
        distrib - the distribution to use
                  (a scipy.stats frozen distribution object)
        input   - the points at which to compute the pdf values
                  (a numpy.ndarray object)
        inpbdf  - the undefined value for input

    Returns:
        None

    Raises:
        ValueError or AttributeError if arguments are not valid
    """
    badmask = ( numpy.fabs(input - inpbdf) < 1.0E-5 )
    goodmask = numpy.logical_not(badmask)
    result[badmask] = resbdf
    # array[goodmask] is a flattened array
    result[goodmask] = distrib.pdf(input[goodmask])


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # Test the distribution name and parameters given to getdistrib give
    # the expected mean, variance, skew, and kurtosis values.  Testing of 
    # assignpdf is done by statspdf.py.

    # Normal distribution
    mu = 5.0
    sigma = 3.0
    distname = "norm"
    distparms = numpy.array([mu, sigma], dtype=numpy.float32)
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( mu, sigma * sigma, 0.0, 0.0 )
    if not numpy.allclose(foundstats, expectedstats):
        raise ValueError("(mean, var, skew, kurtosis) of norm(%.1f, %.1f): expected %s; found %s" % \
                          (mu, sigma, str(expectedstats), str(foundstats)))

    # Chi-squared distribution
    lambdastr = "10"
    distname = "chi-square"
    distparms = [ lambdastr ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    lambdaflt = float(lambdastr)
    expectedstats = ( lambdaflt, 
                      2.0 * lambdaflt, 
                      math.sqrt(8.0 / lambdaflt), 
                      12.0 / lambdaflt )
    if not numpy.allclose(foundstats, expectedstats):
        raise ValueError("(mean, var, skew, kurtosis) of %s(%s): expected %s; found %s" % \
                          (distname, lambdastr, str(expectedstats), str(foundstats)))

    # All successful
    print "Success"

