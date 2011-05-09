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
    distrib = None
    if (lcdistname == "norm") or (lcdistname == "normal"):
        mu = float(distribparams[0])
        sigma = float(distribparams[1])
        distrib = scipy.stats.norm(mu, sigma)
    elif (lcdistname == "chi2") or (lcdistname == "chi-square"):
        degfree = float(distribparams[0])
        distrib = scipy.stats.chi2(degfree)
    elif lcdistname == "poisson":
        lmbda = float(distribparams[0])
        distrib = scipy.stats.poisson(lmbda)
    else:
        # for working with for now: 1 == norm, 2 == chi2, 3 == poisson
        try:
            distnum = int(distribname)
            if distnum == 1:
                mu = float(distribparams[0])
                sigma = float(distribparams[1])
                distrib = scipy.stats.norm(mu, sigma)
            elif distnum == 2:
                degfree = float(distribparams[0])
                distrib = scipy.stats.chi2(degfree)
            elif distnum == 3:
                lmbda = float(distribparams[0])
                distrib = scipy.stats.poisson(lmbda)
        except:
            pass
    if distrib == None:
        raise ValueError, "Unknown probability function %s" % str(distribname)
    return distrib


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # Normal distribution
    mu = 5.0
    sigma = 3.0
    distname = numpy.array([1.0], dtype=numpy.float32)
    distparms = numpy.array([mu, sigma], dtype=numpy.float32)
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( mu, sigma * sigma, 0.0, 0.0 )
    if not numpy.allclose(foundstats, expectedstats):
        raise ValueError, "(mean, var, skew, kurtosis) of norm(%.1f, %.1f): expected %s; found %s" % \
                           (mu, sigma, str(expectedstats), str(foundstats))

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
        raise ValueError, "(mean, var, skew, kurtosis) of %s(%s): expected %s; found %s" % \
                           (distname, lambdastr, str(expectedstats), str(foundstats))

    # All successful
    print "Success"

