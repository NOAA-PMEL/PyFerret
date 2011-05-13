"""
Helper functions for pyferret stats external functions.
"""
import math
import numpy
import scipy.stats
import scipy.special


def getdistrib(distribname, distribparams):
    """
    Creates and returns scipy.stats "frozen" probability distribution
    object.  Converts the "standard" parameters (inluding ordering) for
    a distribution in order to appropriately call the constructor for
    the scipy.stats frozen distribution object.

    If distribparams is None, this instead returns a tuple of (param_name,
    param_descrip) string pairs.  The value param_name is the parameter
    name and the value "param_descript" is a description of the parameter.

    Arguments:
       distribname - name of the distribution
       distribparams - tuple/list/array of input parameters

    Returns:
       the scipy.stats "frozen" distribution object; or if distribparams
       is None, a tuple of (param_name, param_descript) string pairs.

    Raises:
       ValueError if the distribution name is not recognized by this routine,
                  if the incorrect number of parameters are given, or
                  if the distribution parameters are invalid
    """
    lcdistname = str(distribname).lower()
    distrib = None
    if lcdistname == "beta":
        if distribparams == None:
            distrib = ( ( "alpha", "first shape value", ),
                        ( "beta", "second shape value", ), )
        else:
            if len(distribparams) != 2:
                raise ValueError("Two parameters expected for the Beta distribution")
            alpha = float(distribparams[0])
            beta = float(distribparams[1])
            if (alpha <= 0.0) or (beta <= 0.0):
                raise ValueError("Invalid parameter(s) for the Beta distribution")
            distrib = scipy.stats.beta(alpha, beta)
    elif (lcdistname == "binom") or (lcdistname == "binomial"):
        if distribparams == None:
            distrib = ( ( "n", "number of trials", ),
                        ( "p", "success probability in each trial", ), )
        else:
            if len(distribparams) != 2:
                raise ValueError("Two parameters expected for the Binomial distribution")
            nflt = float(distribparams[0])
            prob = float(distribparams[1])
            if (nflt < 0.0) or (prob < 0.0) or (prob > 1.0):
                raise ValueError("Invalid parameter(s) for the Binomial distribution")
            distrib = scipy.stats.binom(nflt, prob)
    elif lcdistname == "cauchy":
        if distribparams == None:
            distrib = ( ( "m", "location (median)", ),
                        ( "gamma", "scale (half-width at half-maximum)", ), )
        else:
            if len(distribparams) != 2:
                raise ValueError("Two parameters expected for the Cauchy distribution")
            m = float(distribparams[0])
            gamma = float(distribparams[1])
            if gamma <= 0.0:
                raise ValueError("Invalid parameter for the Cauchy distribution")
            distrib = scipy.stats.cauchy(m, gamma)
    elif (lcdistname == "chi2") or (lcdistname == "chi-square"):
        if distribparams == None:
            distrib = ( ( "df", "degrees of freedom", ), )
        else:
            if len(distribparams) != 1:
                raise ValueError("One parameter expected for the Chi-squared distribution")
            degfree = float(distribparams[0])
            if degfree <= 0.0:
                raise ValueError("Invalid parameter for the Chi-squared distribution")
            distrib = scipy.stats.chi2(degfree)
    elif (lcdistname == "expon") or (lcdistname == "exponential"):
        if distribparams == None:
            distrib = ( ( "lambda", "rate (inverse scale)", ), )
        else:
            if len(distribparams) != 1:
                raise ValueError("One parameter expected for the Exponential distribution")
            lambdaflt = float(distribparams[0])
            if lambdaflt <= 0.0:
                raise ValueError("Invalid parameter for the Exponential distribution")
            distrib = scipy.stats.expon(scale=(1.0/lambdaflt))
    elif (lcdistname == "exponweib") or (lcdistname == "exponentiated-weibull"):
        if distribparams == None:
            distrib = ( ( "k", "first shape", ),
                        ( "lambda", "scale", ),
                        ( "alpha", "second shape (exponential power)", ), )
        else:
            if len(distribparams) != 3:
                raise ValueError("Three parameters expected for the Exponentiated-weibull distribution")
            k =  float(distribparams[0])
            lambdaflt = float(distribparams[1])
            alpha = float(distribparams[2])
            if (k <= 0.0) or (lambdaflt <= 0.0) or (alpha <= 0):
                raise ValueError("Invalid parameter(s) for the Exponentiated-weibull distribution")
            distrib = scipy.stats.exponweib(alpha, k, scale=lambdaflt)
    elif (lcdistname == "f") or (lcdistname == "fisher"):
        if distribparams == None:
            distrib = ( ( "dfn", "numerator degrees of freedom", ),
                        ( "dfd", "denominator degrees of freedom", ), )
        else:
            if len(distribparams) != 2:
               raise ValueError("Two parameters expected for the F distribution")
            dfnum = float(distribparams[0])
            dfdenom = float(distribparams[1])
            if (dfnum <= 0.0) or (dfdenom <= 0.0):
               raise ValueError("Invalid parameter(s) for the F distribution")
            distrib = scipy.stats.f(dfnum, dfdenom)
    elif lcdistname == "gamma":
        if distribparams == None:
            distrib = ( ( "alpha", "shape parameter", ),
                        ( "theta", "scale parameter", ), )
        else:
            if len(distribparams) != 2:
                raise ValueError("Two parameters expected for the Gamma distribution")
            alpha = float(distribparams[0])
            theta = float(distribparams[1])
            if (alpha <= 0.0) or (theta <= 0.0):
                raise ValueError("Invalid parameter(s) for the Gamma distribution")
            distrib = scipy.stats.gamma(alpha, scale=theta)
    elif (lcdistname == "geom") or (lcdistname == "geometric") or (lcdistname == "shifted-geometric"):
        if distribparams == None:
            distrib = ( ( "p", "success probability", ), )
        else:
            if len(distribparams) != 1:
                raise ValueError("One parameter expected for the Shifted-geometric distribution")
            prob = float(distribparams[0])
            if (prob < 0.0) or (prob > 1.0):
                raise ValueError("Invalid parameter for the Shifted-geometric distribution")
            distrib = scipy.stats.geom(prob)
    elif (lcdistname == "hypergeom") or (lcdistname == "hypergeometric"):
        if distribparams == None:
            distrib = ( ( "ntotal", "total number of items", ),
                        ( "ngood", "total number of 'success' items", ),
                        ( "ndrawn", "number of items selected", ), )
        else:
            if len(distribparams) != 3:
               raise ValueError("Three parameters expected for the Hypergeometric distribution")
            numtotal = float(distribparams[0])
            numgood = float(distribparams[1])
            numdrawn = float(distribparams[2])
            if (numtotal <= 0.0) or (numgood < 0.0) or (numdrawn < 0.0):
               raise ValueError("Invalid parameter(s) for the Hypergeometric distribution")
            distrib = scipy.stats.hypergeom(numtotal, numgood, numdrawn)
    elif lcdistname == "laplace":
        if distribparams == None:
            distrib = ( ( "mu", "location (mean)", ),
                        ( "b", "scale", ), )
        else:
            if len(distribparams) != 2:
                raise ValueError("Two parameters expected for the Laplace distribution")
            mu = float(distribparams[0])
            b = float(distribparams[1])
            if b <= 0.0:
                raise ValueError("Invalid parameter for the Laplace distribution")
            distrib = scipy.stats.laplace(mu, b)
    elif (lcdistname == "lognorm") or (lcdistname == "log-normal"):
        if distribparams == None:
            distrib = ( ( "mu", "log-scale (mean of the natural log of the distribution)", ),
                        ( "sigma", "shape (std. dev. of the natural log of the distribution)", ), )
        else:
            if len(distribparams) != 2:
                raise ValueError("Two parameters expected for the Log-normal distribution")
            mu = math.exp(float(distribparams[0]))
            sigma = float(distribparams[1])
            if sigma <= 0.0:
                raise ValueError("Invalid parameter for the Log-normal distribution")
            distrib = scipy.stats.lognorm(sigma, scale=mu)
    elif (lcdistname == "nbinom") or (lcdistname == "negative-binomial"):
        if distribparams == None:
            distrib = ( ( "n", "number of successes to stop", ),
                        ( "p", "success probability in each trial", ), )
        else:
            if len(distribparams) != 2:
                raise ValueError("Two parameters expected for the Negative-binomial distribution")
            numsuccess = float(distribparams[0])
            prob = float(distribparams[1])
            if (numsuccess < 1.0) or (prob <= 0.0) or (prob > 1.0):
                raise ValueError("Invalid parameter(s) for the Negative-binomial distribution")
            distrib = scipy.stats.nbinom(numsuccess, prob)
    elif (lcdistname == "norm") or (lcdistname == "normal"):
        if distribparams == None:
            distrib = ( ( "mu", "mean", ),
                        ( "sigma", "standard deviation", ), )
        else:
            if len(distribparams) != 2:
                raise ValueError("Two parameters expected for the Normal distribution")
            mu = float(distribparams[0])
            sigma = float(distribparams[1])
            if sigma <= 0.0:
                raise ValueError("Invalid parameter for the Normal distribution")
            distrib = scipy.stats.norm(mu, sigma)
    elif lcdistname == "pareto":
        if distribparams == None:
            distrib = ( ( "xm", "scale (minimum abscissa value)", ),
                        ( "alpha", "shape", ), )
        else:
            if len(distribparams) != 2:
                raise ValueError("Two parameters expected for the Pareto distribution")
            xm =  float(distribparams[0])
            alpha = float(distribparams[1])
            if (xm <= 0.0) or (alpha <= 0.0):
                raise ValueError("Invalid parameter(s) for the Pareto distribution")
            distrib = scipy.stats.pareto(alpha, scale=xm)
    elif lcdistname == "poisson":
        if distribparams == None:
            distrib = ( ( "mu", "expected number of occurences", ), )
        else:
            if len(distribparams) != 1:
                raise ValueError("One parameter expected for the Poisson distribution")
            mu = float(distribparams[0])
            if mu <= 0.0:
                raise ValueError("Invalid parameter for the Poisson distribution")
            distrib = scipy.stats.poisson(mu)
    elif (lcdistname == "t") or (lcdistname == "students-t"):
        if distribparams == None:
            distrib = ( ( "df", "degrees of freedom", ), )
        else:
            if len(distribparams) != 1:
                raise ValueError("One parameter expected for the Student-t distribution")
            degfree = float(distribparams[0])
            if degfree <= 0.0:
                raise ValueError("Invalid parameter for the Student-t distribution")
            distrib = scipy.stats.t(degfree)
    elif (lcdistname == "weibull_min") or (lcdistname == "weibull"):
        if distribparams == None:
            distrib = ( ( "k", "shape", ),
                        ( "lambda", "scale", ), )
        else:
            if len(distribparams) != 2:
                raise ValueError("Two parameters expected for the Weibull distribution")
            k =  float(distribparams[0])
            lambdaflt = float(distribparams[1])
            if (k <= 0.0) or (lambdaflt <= 0.0):
                raise ValueError("Invalid parameter(s) for the Weibull distribution")
            distrib = scipy.stats.weibull_min(k, scale=lambdaflt)
    else:
        raise ValueError("Unknown probability function %s" % str(distribname))
    if distrib == None:
        raise ValueError("Problems obtaining the probability distribution object")
    return distrib


def assignpdf(result, resbdf, distrib, input, inpbdf):
    """
    Assigns the probability density function values of a continuous
    distribution at specified positions.  At undefined positions, the
    results will be assigned as undefined.

    Arguments:
        result  - the numpy.ndarray to be assigned the pdf values
        resbdf  - the undefined value for result
        distrib - the continuous distribution to use
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
    badmask = numpy.logical_or(badmask, numpy.isnan(input))
    goodmask = numpy.logical_not(badmask)
    result[badmask] = resbdf
    # array[goodmask] is a flattened array
    result[goodmask] = distrib.pdf(input[goodmask])


def assignpmf(result, resbdf, distrib, input, inpbdf):
    """
    Assigns the probability mass function values of a discrete distribution
    at specified positions.  At undefined positions, the results will be
    assigned as undefined.

    Arguments:
        result  - the numpy.ndarray to be assigned the pmf values
        resbdf  - the undefined value for result
        distrib - the discrete distribution to use
                  (a scipy.stats frozen distribution object)
        input   - the points at which to compute the pmf values
                  (a numpy.ndarray object)
        inpbdf  - the undefined value for input

    Returns:
        None

    Raises:
        ValueError or AttributeError if arguments are not valid
    """
    badmask = ( numpy.fabs(input - inpbdf) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(input))
    goodmask = numpy.logical_not(badmask)
    result[badmask] = resbdf
    # array[goodmask] is a flattened array
    result[goodmask] = distrib.pmf(input[goodmask])


def assigncdf(result, resbdf, distrib, input, inpbdf):
    """
    Assigns the cumulative distribution function values of a distribution
    at specified positions.  At undefined positions, the results will be
    assigned as undefined.

    Arguments:
        result  - the numpy.ndarray to be assigned the cdf values
        resbdf  - the undefined value for result
        distrib - the distribution to use
                  (a scipy.stats frozen distribution object)
        input   - the points at which to compute the cdf values
                  (a numpy.ndarray object)
        inpbdf  - the undefined value for input

    Returns:
        None

    Raises:
        ValueError or AttributeError if arguments are not valid
    """
    badmask = ( numpy.fabs(input - inpbdf) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(input))
    goodmask = numpy.logical_not(badmask)
    result[badmask] = resbdf
    # array[goodmask] is a flattened array
    result[goodmask] = distrib.cdf(input[goodmask])


def assignsf(result, resbdf, distrib, input, inpbdf):
    """
    Assigns the suvival function values of a distribution at specified
    positions.  At undefined positions, the results will be assigned as
    undefined.

    Arguments:
        result  - the numpy.ndarray to be assigned the sf values
        resbdf  - the undefined value for result
        distrib - the distribution to use
                  (a scipy.stats frozen distribution object)
        input   - the points at which to compute the sf values
                  (a numpy.ndarray object)
        inpbdf  - the undefined value for input

    Returns:
        None

    Raises:
        ValueError or AttributeError if arguments are not valid
    """
    badmask = ( numpy.fabs(input - inpbdf) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(input))
    goodmask = numpy.logical_not(badmask)
    result[badmask] = resbdf
    # array[goodmask] is a flattened array
    result[goodmask] = distrib.sf(input[goodmask])


def assignppf(result, resbdf, distrib, input, inpbdf):
    """
    Assigns the percent point function values of a distribution at
    specified positions.  At undefined positions, the results will
    be assigned as undefined.

    Arguments:
        result  - the numpy.ndarray to be assigned the ppf values
        resbdf  - the undefined value for result
        distrib - the distribution to use
                  (a scipy.stats frozen distribution object)
        input   - the points at which to compute the ppf values
                  (a numpy.ndarray object)
        inpbdf  - the undefined value for input

    Returns:
        None

    Raises:
        ValueError or AttributeError if arguments are not valid
    """
    badmask = ( numpy.fabs(input - inpbdf) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(input))
    goodmask = numpy.logical_not(badmask)
    result[badmask] = resbdf
    # array[goodmask] is a flattened array
    result[goodmask] = distrib.ppf(input[goodmask])


def assignisf(result, resbdf, distrib, input, inpbdf):
    """
    Assigns the inverse survival function values of a distribution at
    specified positions.  At undefined positions, the results will be
    assigned as undefined.

    Arguments:
        result  - the numpy.ndarray to be assigned the isf values
        resbdf  - the undefined value for result
        distrib - the distribution to use
                  (a scipy.stats frozen distribution object)
        input   - the points at which to compute the isf values
                  (a numpy.ndarray object)
        inpbdf  - the undefined value for input

    Returns:
        None

    Raises:
        ValueError or AttributeError if arguments are not valid
    """
    badmask = ( numpy.fabs(input - inpbdf) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(input))
    goodmask = numpy.logical_not(badmask)
    result[badmask] = resbdf
    # array[goodmask] is a flattened array
    result[goodmask] = distrib.isf(input[goodmask])


def assignrvs(result, resbdf, distrib, input, inpbdf):
    """
    Assigns the random variates of a distribution at positions in the
    result array corresponding to defined values in the input array.
    At undefined positions in the input array, the results array value
    will be undefined.

    Arguments:
        result  - the numpy.ndarray to be assigned the random variates
        resbdf  - the undefined value for result
        distrib - the distribution to use
                  (a scipy.stats frozen distribution object)
        input   - the input array indicating positions to be assigned
                  (a numpy.ndarray object)
        inpbdf  - the undefined value for input

    Returns:
        None

    Raises:
        ValueError or AttributeError if arguments are not valid
    """
    badmask = ( numpy.fabs(input - inpbdf) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(input))
    goodmask = numpy.logical_not(badmask)
    result[badmask] = resbdf
    # result[goodmask] is a flattened array
    result[goodmask] = distrib.rvs(len(result[goodmask]))


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # Test the distribution name and parameters given to getdistrib
    # give the expected distribution.  (Primarily that the parameters
    # are interpreted and assigned correctly.)  Testing of the other
    # functions are performed by the stats_*.py scipts.

    # Beta distribution
    distname = "Beta"
    descript = getdistrib(distname, None)
    if len(descript) != 2:
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    alpha = 1.5
    beta = 2.75
    distparms = [ alpha, beta ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( alpha / (alpha + beta),
                      alpha * beta / ((alpha + beta)**2 * (alpha + beta + 1.0)),
                      2.0 * (beta - alpha) / (2.0 + alpha + beta) *
                          math.sqrt((1.0 + alpha + beta) / (alpha * beta)),
                      6.0 * (alpha**3 + alpha**2 * (1.0 - 2.0 * beta) + \
                          beta**2 * (1.0 + beta) - 2.0 * alpha * beta * (2.0 + beta)) / \
                          (alpha * beta * (alpha + beta + 2.0) * (alpha + beta + 3.0)),
                    )
    if not numpy.allclose(foundstats, expectedstats):
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))

    # Binomial distribution
    distname = "binom"
    descript = getdistrib(distname, None)
    if len(descript) != 2:
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    ntrials = 20.0
    prob = 0.25
    distparms = [ ntrials, prob ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( ntrials * prob,
                      ntrials * prob * (1.0 - prob),
                      (1.0 - 2.0 * prob) / math.sqrt(ntrials * prob * (1.0 - prob)),
                      (1.0 - 6.0 * prob * (1.0 - prob)) / (ntrials * prob * (1.0 - prob)),
                    )
    if not numpy.allclose(foundstats, expectedstats):
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))

    # Chi-squared distribution
    distname = "chi2"
    descript = getdistrib(distname, None)
    if len(descript) != 1:
        raise ValueError("number of parameter description pairs for %s: expected 1; found %d:" % \
                         (distname, len(descript)))
    lambdastr = "10"
    distparms = [ lambdastr ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    lambdaflt = float(lambdastr)
    expectedstats = ( lambdaflt,
                      2.0 * lambdaflt,
                      math.sqrt(8.0 / lambdaflt),
                      12.0 / lambdaflt,
                    )
    if not numpy.allclose(foundstats, expectedstats):
        raise ValueError("(mean, var, skew, kurtosis) of %s(%s): expected %s; found %s" % \
                          (distname, distparms[0], str(expectedstats), str(foundstats)))

    # Cauchy distribution
    distname = "cauchy"
    descript = getdistrib(distname, None)
    if len(descript) != 2:
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    m = 5.0
    gamma = 2.0
    distparms = [ m, gamma ]
    distf = getdistrib(distname, distparms)
    # mean, variance, skew, kurtosis undefined; instead check some pdf values
    xvals = numpy.arange(0.0, 10.1, 0.5)
    foundpdfs = distf.pdf(xvals)
    expectedpdfs = (gamma / numpy.pi) / ((xvals - m)**2 + gamma**2)
    if not numpy.allclose(foundpdfs, expectedpdfs):
        raise ValueError("pdfs(0.0:10.1:0.5) of %s(%#.1f,%#.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedpdfs), str(foundpdfs)))

    # Exponential distribution
    distname = "expon"
    descript = getdistrib(distname, None)
    if len(descript) != 1:
        raise ValueError("number of parameter description pairs for %s: expected 1; found %d:" % \
                         (distname, len(descript)))
    lambdaflt = 11.0
    distparms = [ lambdaflt ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( 1.0 / lambdaflt, 1.0 / lambdaflt**2, 2.0, 6.0 )
    if not numpy.allclose(foundstats, expectedstats):
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f): expected %s; found %s" % \
                          (distname, distparms[0], str(expectedstats), str(foundstats)))

    # Exponentiated Weibull distribution
    distname = "exponweib"
    descript = getdistrib(distname, None)
    if len(descript) != 3:
        raise ValueError("number of parameter description pairs for %s: expected 3; found %d:" % \
                         (distname, len(descript)))
    k = 3.0
    lambdaflt = 5.0
    alpha = 2.5
    distparms = [ k, lambdaflt, alpha ]
    distf = getdistrib(distname, distparms)
    # don't know the formula for the mean, variance, skew, kurtosis
    # instead check some cdf values
    xvals = numpy.arange(0.0, 10.1, 0.5)
    foundcdfs = distf.cdf(xvals)
    expectedcdfs = numpy.power(1.0 - numpy.exp(-1.0 * numpy.power(xvals / lambdaflt, k)), alpha)
    if not numpy.allclose(foundcdfs, expectedcdfs):
        raise ValueError("cdfs(0.0:10.1:0.5) of %s(%#.1f,%#.1f%#.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], distparms[2], str(expectedcdfs), str(foundcdfs)))

    # F distribution
    distname = "F"
    descript = getdistrib(distname, None)
    if len(descript) != 2:
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    dofn = 7.0
    dofd = 11.0   # needs to be larger than 8.0 for kurtosis formula
    distparms = [ dofn, dofd ]
    distf = getdistrib(distname, distparms)
    # foundstats = distf.stats("mvsk")
    foundstats = distf.stats("mv")
    expectedstats = ( dofd / (dofd - 2.0),
                      2.0 * dofd**2 * (dofn + dofd - 2.0) / \
                          (dofn * (dofd - 2.0)**2 * (dofd - 4.0)),
                      # ((2.0 * dofn + dofd - 2.0) / (dofd - 6.0)) * \
                      #     math.sqrt(8.0 * (dofd - 4.0) / (dofn * (dofn + dofd - 2.0))),
                      # 12.0 * (20.0 * dofd - 8.0 * dofd**2 + dofd**3 + 44.0 * dofn - 32.0 * dofn * dofd + \
                      #     5.0 * dofd**2 * dofn - 22.0 * dofn**2 + 5.0 * dofd * dofn**2 - 16.0) / \
                      #     (dofn * (dofd - 6.0) * (dofd - 8.0) * (dofn + dofd - 2)),
                    )
    if not numpy.allclose(foundstats, expectedstats):
        # raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
        raise ValueError("(mean, var) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))
    # since skew and kurtosis is not coming out as expected, check some pdf values
    xvals = numpy.arange(0.0, 10.1, 0.5)
    foundpdfs = distf.pdf(xvals)
    factor = scipy.special.gamma(0.5 * (dofn + dofd)) / \
             (scipy.special.gamma(0.5 * dofn) * scipy.special.gamma(0.5 *dofd))
    factor *= math.pow(dofn, 0.5 * dofn) * math.pow(dofd, 0.5 * dofd)
    expectedpdfs = factor * numpy.power(xvals, 0.5 * dofn - 1.0) / \
                   numpy.power(dofd + dofn * xvals, 0.5 * (dofn + dofd))
    if not numpy.allclose(foundpdfs, expectedpdfs):
        raise ValueError("pdfs(0.0:10.1:0.5) of %s(%#.1f,%#.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedpdfs), str(foundpdfs)))

    # Gamma distribution
    distname = "gamma"
    descript = getdistrib(distname, None)
    if len(descript) != 2:
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    alpha = 5.0
    theta = 3.0
    distparms = [ alpha, theta ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( alpha * theta, alpha * theta**2, 2.0 / math.sqrt(alpha), 6.0 / alpha )
    if not numpy.allclose(foundstats, expectedstats):
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))

    # Geometric distribution
    distname = "geometric"
    descript = getdistrib(distname, None)
    if len(descript) != 1:
        raise ValueError("number of parameter description pairs for %s: expected 1; found %d:" % \
                         (distname, len(descript)))
    prob = 0.25
    distparms = [ prob ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( 1.0 / prob,
                     (1.0 - prob) / prob**2,
                     (2.0 - prob) / math.sqrt(1.0 - prob),
                     6.0 + prob**2 / (1.0 - prob),
                    )
    if not numpy.allclose(foundstats, expectedstats):
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f): expected %s; found %s" % \
                          (distname, distparms[0], str(expectedstats), str(foundstats)))

    # Hypergeometric distribution
    distname = "hypergeom"
    descript = getdistrib(distname, None)
    if len(descript) != 3:
        raise ValueError("number of parameter description pairs for %s: expected 3; found %d:" % \
                         (distname, len(descript)))
    numtotal = 29.0
    numgood = 13.0
    numdrawn = 17.0
    distparms = [ numtotal, numgood, numdrawn ]
    distf = getdistrib(distname, distparms)
    # foundstats = distf.stats("mvsk")
    foundstats = distf.stats("mvs")
    expectedstats = ( numdrawn * numgood / numtotal,
                      numdrawn * numgood * (numtotal - numdrawn) * (numtotal - numgood) / \
                          (numtotal**2 * (numtotal - 1.0)),
                      math.sqrt(numtotal - 1.0) * (numtotal - 2.0 * numdrawn) * (numtotal - 2.0 * numgood) / \
                          (math.sqrt(numdrawn * numgood * (numtotal - numdrawn) * (numtotal - numgood)) * \
                              (numtotal - 2.0)),
                      # (numtotal**2 * (numtotal - 1.0) / \
                      #         (numdrawn * (numtotal - 2.0) * (numtotal - 3.0) * (numtotal - numdrawn))) * \
                      #     ((numtotal * (numtotal + 1.0) - 6.0 * numtotal * (numtotal - numdrawn)) / \
                      #      (numgood * (numtotal - numgood)) + \
                      #      3.0 * numdrawn * (numtotal - numdrawn) * (numtotal + 6.0) / numtotal**2 - 6.0),
                    )
    if not numpy.allclose(foundstats, expectedstats):
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], distparms[2], str(expectedstats), str(foundstats)))

    # Laplace distribution
    distname = "laplace"
    descript = getdistrib(distname, None)
    if len(descript) != 2:
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    mu = 5.0
    b = 3.0
    distparms = [ mu, b ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( mu, 2.0 * b**2, 0.0, 3.0 )
    if not numpy.allclose(foundstats, expectedstats):
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))

    # Log-normal distribution
    distname = "lognorm"
    descript = getdistrib(distname, None)
    if len(descript) != 2:
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    mu = 0.8
    sigma = 0.5
    distparms = numpy.array([ mu, sigma ], dtype=numpy.float32)
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( math.exp(mu + 0.5 * sigma**2),
                      math.exp(2.0 * mu + sigma**2) * (math.exp(sigma**2) - 1.0),
                      (2.0 + math.exp(sigma**2)) * math.sqrt(math.exp(sigma**2) - 1.0),
                      math.exp(4.0 * sigma**2) + 2.0 * math.exp(3.0 * sigma**2) + 3.0 * math.exp(2.0 * sigma**2) - 6,
                    )
    if not numpy.allclose(foundstats, expectedstats):
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))

    # Negative-binomial distribution
    distname = "nbinom"
    descript = getdistrib(distname, None)
    if len(descript) != 2:
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    numsuccess = 5.0
    prob = 0.25
    distparms = [ numsuccess, prob ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( numsuccess * (1.0 - prob) / prob,
                      numsuccess * (1.0 - prob) / prob**2,
                      (2.0 - prob) / math.sqrt(numsuccess * (1.0 - prob)),
                      (prob**2 - 6.0 * prob + 6.0) / (numsuccess * (1.0 - prob)),
                    )
    if not numpy.allclose(foundstats, expectedstats):
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))

    # Normal distribution
    distname = "norm"
    descript = getdistrib(distname, None)
    if len(descript) != 2:
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    mu = 5.0
    sigma = 3.0
    distparms = numpy.array([ mu, sigma ], dtype=numpy.float32)
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( mu, sigma**2, 0.0, 0.0 )
    if not numpy.allclose(foundstats, expectedstats):
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))

    # Pareto distribution
    distname = "pareto"
    descript = getdistrib(distname, None)
    if len(descript) != 2:
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    xm = 3.0
    alpha = 5.0  # must be larger than 4 for kurtosis formula
    distparms = [ xm, alpha ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( alpha * xm / (alpha - 1.0),
                      xm**2 * alpha / ((alpha - 1.0)**2 * (alpha - 2.0)),
                      2.0 * ((alpha + 1.0) / (alpha - 3.0)) * math.sqrt((alpha - 2.0) / alpha),
                      6.0 * (alpha**3 + alpha**2 - 6.0 * alpha - 2.0) / \
                          (alpha * (alpha - 3.0) * (alpha - 4.0)),
                    )
    if not numpy.allclose(foundstats, expectedstats):
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))

    # Poisson distribution
    distname = "poisson"
    descript = getdistrib(distname, None)
    if len(descript) != 1:
        raise ValueError("number of parameter description pairs for %s: expected 1; found %d:" % \
                         (distname, len(descript)))
    mu = 7.0
    distparms = [ mu ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( mu, mu, 1.0 / math.sqrt(mu), 1.0 / mu )
    if not numpy.allclose(foundstats, expectedstats):
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f): expected %s; found %s" % \
                          (distname, distparms[0], str(expectedstats), str(foundstats)))


    # Student's-t distribution
    distname = "t"
    descript = getdistrib(distname, None)
    if len(descript) != 1:
        raise ValueError("number of parameter description pairs for %s: expected 1; found %d:" % \
                         (distname, len(descript)))
    degfree = 11.0
    distparms = [ degfree ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( 0.0, degfree / (degfree - 2.0), 0.0, 6.0 / (degfree - 4.0) )
    if not numpy.allclose(foundstats, expectedstats):
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f): expected %s; found %s" % \
                          (distname, distparms[0], str(expectedstats), str(foundstats)))

    # Weibull distribution
    distname = "weibull"
    descript = getdistrib(distname, None)
    if len(descript) != 2:
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    k = 3.0
    lambdaflt = 5.0
    distparms = [ k, lambdaflt ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    gam1 = scipy.special.gamma(1.0 + 1.0 / k)
    gam2 = scipy.special.gamma(1.0 + 2.0 / k)
    gam3 = scipy.special.gamma(1.0 + 3.0 / k)
    gam4 = scipy.special.gamma(1.0 + 4.0 / k)
    mu = lambdaflt * gam1
    sigma = lambdaflt * math.sqrt(gam2 - gam1**2)
    expectedstats = ( mu,
                      sigma**2,
                      (lambdaflt**3 * gam3  - 3.0 * mu * sigma**2 - mu**3) / sigma**3,
                      (gam4 - 4.0 * gam1 * gam3 - 3.0 * gam2**2 + 12.0 * gam1**2 * gam2 - 6.0 * gam1**4) / \
                      (gam2 - gam1**2)**2,
                    )
    if not numpy.allclose(foundstats, expectedstats):
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))

    # All successful
    print "Success"

