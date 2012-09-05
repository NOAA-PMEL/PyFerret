"""
Helper functions for pyferret stats external functions.
"""
import math
import numpy
try:
    import scipy.stats
    import scipy.special
except ImportError:
    # do not raise an error until actually used
    pass
import pyferret


def getdistname(distribname=None):
    """
    Translates probability distribution names into scipy.stats names.
    Arguments:
        distribname - the distribution name, or None.
    Returns:
        If distribname is given (and not None), the "scipy.stats" name for
            the probability distribution given in distribname, or None if
            the probability distribution name is not recognized or supported.
        If distribname is not given (or None), returns a list of string
            tuples, where the first name in each tuple is the scipy.stats
            name, the second name is a "full name" and any other names are
            other recognized aliases.
    """
    namelist = (
                 ( "beta", "Beta", ),
                 ( "binom", "Binomial", ),
                 ( "cauchy", "Cauchy", ),
                 ( "chi", "Chi", ),
                 ( "chi2", "Chi-Square", ),
                 ( "expon", "Exponential", ),
                 ( "exponweib", "Exponentiated-Weibull", ),
                 ( "f", "F", "Fisher", ),
                 ( "gamma", "Gamma", ),
                 ( "geom", "Geometric", "Shifted-Geometric", ),
                 ( "hypergeom", "Hypergeometric", ),
                 ( "invgamma", "Inverse-Gamma", ),
                 ( "laplace", "Laplace", ),
                 ( "lognorm", "Log-Normal", ),
                 ( "nbinom", "Negative-Binomial", ),
                 ( "norm", "Normal", ),
                 ( "pareto", "Pareto", ),
                 ( "poisson", "Poisson", ),
                 ( "randint", "Random-Integer", "Discrete-Uniform", ),
                 ( "t", "Students-T", ),
                 ( "uniform", "Uniform", ),
                 ( "weibull_min", "Weibull", ),
               )
    if distribname == None:
        return namelist
    lcdistname = str(distribname).lower()
    # Testing below verifies the above names are all recognized in the following
    if lcdistname == "beta":
        return "beta"
    if (lcdistname == "binom") or (lcdistname == "binomial"):
        return "binom"
    if lcdistname == "cauchy":
        return "cauchy"
    if lcdistname == "chi":
        return "chi"
    if (lcdistname == "chi2") or (lcdistname == "chi-square"):
        return "chi2"
    if (lcdistname == "expon") or (lcdistname == "exponential"):
        return "expon"
    if (lcdistname == "exponweib") or (lcdistname == "exponentiated-weibull"):
        return "exponweib"
    if (lcdistname == "f") or (lcdistname == "fisher"):
        return "f"
    if lcdistname == "gamma":
        return "gamma"
    if (lcdistname == "geom") or (lcdistname == "geometric") or (lcdistname == "shifted-geometric"):
        return "geom"
    if (lcdistname == "hypergeom") or (lcdistname == "hypergeometric"):
        return "hypergeom"
    if (lcdistname == "invgamma") or (lcdistname == "inverse-gamma"):
        return "invgamma"
    if lcdistname == "laplace":
        return "laplace"
    if (lcdistname == "lognorm") or (lcdistname == "log-normal"):
        return "lognorm"
    if (lcdistname == "nbinom") or (lcdistname == "negative-binomial"):
        return "nbinom"
    if (lcdistname == "norm") or (lcdistname == "normal"):
        return "norm"
    if lcdistname == "pareto":
        return "pareto"
    if lcdistname == "poisson":
        return "poisson"
    if (lcdistname == "randint") or (lcdistname == "random-integer") or (lcdistname == "discrete-uniform"):
        return "randint"
    if (lcdistname == "t") or (lcdistname == "students-t"):
        return "t"
    if lcdistname == "uniform":
        return "uniform"
    if (lcdistname == "weibull_min") or (lcdistname == "weibull"):
        return "weibull_min"
    return None


def getdistparams(distname, params, tostd=False):
    """
    Converts between "standard" parameters for a probability function and
    the tuple of numeric parameters expected for the scipy.stats function.

    Arguments:
        distname - the "scipy.stats" name for the probability distribution
        params - the parameters for the probability distribution, or None
        tostd - convert from scipy.stats to standard parameters?

    Returns:
        If params is None, a list of string pairs describing the standard
        parameters for the distribution.

        If tostd is False, a tuple of scipy.stats parameters corresponding
        to the standard parameters given in params.  The given parameters
        are explicitly converted to the correct type (thus strings could
        be given) and error checking is performed.  Additional offset and
        scaling parameters, if appropriate, can be given.

        If tostd is True, a tuple of standard parameters corresponding
        to the scipy.stats parameters given in params.  No explicit type
        conversion or error checking is performed.  Offset and scaling
        parameters, if appropriate, are assumed to be given.

    Raises:
        ValueError if the parameters are inappropriate for the probabiity
        distribution.
    """
    if distname == "beta":
        if params == None:
            return ( ( "ALPHA", "first shape", ),
                     ( "BETA", "second shape", ), )
        if tostd:
            return ( params[0], params[1], params[2], params[3], )
        if (len(params) < 2) or (len(params) > 4):
            raise ValueError("Two to four parameters expected for the Beta distribution")
        alpha = float(params[0])
        beta = float(params[1])
        if (alpha <= 0.0) or (beta <= 0.0):
            raise ValueError("Invalid parameter(s) for the Beta distribution")
        offset = 0.0
        scaling = 1.0
        try:
            offset = float(params[2])
            scaling = float(params[3])
        except IndexError:
            pass
        return ( alpha, beta, offset, scaling, )
    if distname == "binom":
        if params == None:
            return ( ( "N", "number of trials", ),
                     ( "P", "success probability in each trial", ), )
        if tostd:
            return ( params[0], params[1], )
        if len(params) != 2:
            raise ValueError("Two parameters expected for the Binomial distribution")
        nflt = float(params[0])
        prob = float(params[1])
        if (nflt < 0.0) or (prob < 0.0) or (prob > 1.0):
            raise ValueError("Invalid parameter(s) for the Binomial distribution")
        return ( nflt, prob, )
    if distname == "cauchy":
        if params == None:
            return ( ( "M", "location (median)", ),
                     ( "GAMMA", "scale (half-width at half-maximum)", ), )
        if tostd:
            return ( params[0], params[1], )
        if len(params) != 2:
            raise ValueError("Two parameters expected for the Cauchy distribution")
        m = float(params[0])
        gamma = float(params[1])
        if gamma <= 0.0:
            raise ValueError("Invalid parameter for the Cauchy distribution")
        return ( m, gamma, )
    if distname == "chi":
        if params == None:
            return ( ( "DF", "degrees of freedom", ), )
        if tostd:
            return ( params[0], params[1], params[2], )
        if (len(params) < 1) or (len(params) > 3):
            raise ValueError("One to three parameters expected for the Chi distribution")
        degfree = float(params[0])
        if degfree <= 0.0:
            raise ValueError("Invalid parameter for the Chi distribution")
        offset = 0.0
        scaling = 1.0
        try:
            offset = float(params[1])
            scaling = float(params[2])
        except IndexError:
            pass
        return ( degfree, offset, scaling, )
    if distname == "chi2":
        if params == None:
            return ( ( "DF", "degrees of freedom", ), )
        if tostd:
            return ( params[0], params[1], params[2], )
        if (len(params) < 1) or (len(params) > 3):
            raise ValueError("One to three parameters expected for the Chi-Square distribution")
        degfree = float(params[0])
        if degfree <= 0.0:
            raise ValueError("Invalid parameter for the Chi-Square distribution")
        offset = 0.0
        scaling = 1.0
        try:
            offset = float(params[1])
            scaling = float(params[2])
        except IndexError:
            pass
        return ( degfree, offset, scaling, )
    if distname == "expon":
        if params == None:
            return ( ( "LAMBDA", "rate (inverse scale)", ), )
        if tostd:
            return ( 1.0 / params[1], params[0], )
        if (len(params) < 1) or (len(params) > 2):
            raise ValueError("One or two parameters expected for the Exponential distribution")
        lambdaflt = float(params[0])
        if lambdaflt <= 0.0:
            raise ValueError("Invalid parameter for the Exponential distribution")
        try:
            offset = float(params[1])
        except IndexError:
            offset = 0.0
        return ( offset, 1.0 / lambdaflt, )
    if distname == "exponweib":
        if params == None:
            return ( ( "K", "Weibull shape", ),
                     ( "LAMBDA", "scale", ),
                     ( "ALPHA", "power shape", ), )
        if tostd:
            return ( params[1], params[3], params[0], params[2], )
        if (len(params) < 3) or (len(params) > 4):
            raise ValueError("Three or four parameters expected for the Exponentiated-Weibull distribution")
        k =  float(params[0])
        lambdaflt = float(params[1])
        alpha = float(params[2])
        if (k <= 0.0) or (lambdaflt <= 0.0) or (alpha <= 0):
            raise ValueError("Invalid parameter(s) for the Exponentiated-Weibull distribution")
        try:
            offset = float(params[3])
        except IndexError:
            offset = 0.0
        return ( alpha, k, offset, lambdaflt, )
    if distname == "f":
        if params == None:
            return ( ( "DFN", "numerator degrees of freedom", ),
                     ( "DFD", "denominator degrees of freedom", ), )
        if tostd:
            return ( params[0], params[1], params[2], params[3], )
        if (len(params) < 2) or (len(params) > 4):
            raise ValueError("Two to four parameters expected for the F distribution")
        dfnum = float(params[0])
        dfdenom = float(params[1])
        if (dfnum <= 0.0) or (dfdenom <= 0.0):
           raise ValueError("Invalid parameter(s) for the F distribution")
        offset = 0.0
        scaling = 1.0
        try:
            offset = float(params[2])
            scaling = float(params[3])
        except IndexError:
            pass
        return ( dfnum, dfdenom, offset, scaling, )
    if distname == "gamma":
        if params == None:
            return ( ( "ALPHA", "shape", ),
                     ( "THETA", "scale", ), )
        if tostd:
            return ( params[0], params[2], params[1], )
        if (len(params) < 2) or (len(params) > 3):
            raise ValueError("Two or three parameters expected for the Gamma distribution")
        alpha = float(params[0])
        theta = float(params[1])
        if (alpha <= 0.0) or (theta <= 0.0):
            raise ValueError("Invalid parameter(s) for the Gamma distribution")
        try:
            offset = float(params[2])
        except IndexError:
            offset = 0.0
        return ( alpha, offset, theta, )
    if distname == "geom":
        if params == None:
            return ( ( "P", "success probability", ), )
        if tostd:
            return ( params[0], )
        if len(params) != 1:
            raise ValueError("One parameter expected for the Shifted-Geometric distribution")
        prob = float(params[0])
        if (prob < 0.0) or (prob > 1.0):
            raise ValueError("Invalid parameter for the Shifted-Geometric distribution")
        return ( prob, )
    if distname == "hypergeom":
        if params == None:
            return ( ( "NTOTAL", "total number of items", ),
                     ( "NGOOD", "total number of 'success' items", ),
                     ( "NDRAWN", "number of items selected", ), )
        if tostd:
            return ( params[0], params[1], params[2], )
        if len(params) != 3:
           raise ValueError("Three parameters expected for the Hypergeometric distribution")
        numtotal = float(params[0])
        numgood = float(params[1])
        numdrawn = float(params[2])
        if (numtotal <= 0.0) or (numgood < 0.0) or (numdrawn < 0.0):
           raise ValueError("Invalid parameter(s) for the Hypergeometric distribution")
        return ( numtotal, numgood, numdrawn, )
    if distname == "invgamma":
        if params == None:
            return ( ( "ALPHA", "shape", ),
                     ( "BETA", "scale", ), )
        if tostd:
            return ( params[0], params[2], params[1], )
        if (len(params) < 2) or (len(params) > 3):
            raise ValueError("Two or three parameters expected for the Inverse-Gamma distribution")
        alpha = float(params[0])
        beta = float(params[1])
        if (alpha <= 0.0) or (beta <= 0.0):
            raise ValueError("Invalid parameter(s) for the Inverse-Gamma distribution")
        try:
            offset = float(params[2])
        except IndexError:
            offset = 0.0
        return ( alpha, offset, beta, )
    if distname == "laplace":
        if params == None:
            return ( ( "MU", "location (mean)", ),
                     ( "B", "scale", ), )
        if tostd:
            return ( params[0], params[1], )
        if len(params) != 2:
            raise ValueError("Two parameters expected for the Laplace distribution")
        mu = float(params[0])
        b = float(params[1])
        if b <= 0.0:
            raise ValueError("Invalid parameters for the Laplace distribution")
        return ( mu, b, )
    if distname == "lognorm":
        if params == None:
            return ( ( "MU", "log-scale (mean of the natural log of the distribution)", ),
                     ( "SIGMA", "shape (std. dev. of the natural log of the distribution)", ), )
        if tostd:
            return ( math.log(params[2]), params[0], params[1], )
        if (len(params) < 2) or (len(params) > 3):
            raise ValueError("Two or three parameters expected for the Log-Normal distribution")
        mu = math.exp(float(params[0]))
        sigma = float(params[1])
        if sigma <= 0.0:
            raise ValueError("Invalid parameter for the Log-Normal distribution")
        try:
            offset = float(params[2])
        except IndexError:
            offset = 0.0
        return ( sigma, offset, mu, )
    if distname == "nbinom":
        if params == None:
            return ( ( "N", "number of successes to stop", ),
                     ( "P", "success probability in each trial", ), )
        if tostd:
            return ( params[0], params[1], )
        if len(params) != 2:
            raise ValueError("Two parameters expected for the Negative-Binomial distribution")
        numsuccess = float(params[0])
        prob = float(params[1])
        if (numsuccess < 1.0) or (prob <= 0.0) or (prob > 1.0):
            raise ValueError("Invalid parameter(s) for the Negative-Binomial distribution")
        return ( numsuccess, prob, )
    if distname == "norm":
        if params == None:
            return ( ( "MU", "mean value", ),
                     ( "SIGMA", "standard deviation", ), )
        if tostd:
            return ( params[0], params[1], )
        if len(params) != 2:
            raise ValueError("Two parameters expected for the Normal distribution")
        mu = float(params[0])
        sigma = float(params[1])
        if sigma <= 0.0:
            raise ValueError("Invalid parameter for the Normal distribution")
        return ( mu, sigma, )
    if distname == "pareto":
        if params == None:
            return ( ( "XM", "scale (minimum abscissa value)", ),
                     ( "ALPHA", "shape", ), )
        if tostd:
            return ( params[2], params[0], params[1], )
        if (len(params) < 2) or (len(params) > 3):
            raise ValueError("Two or three parameters expected for the Pareto distribution")
        xm =  float(params[0])
        alpha = float(params[1])
        if (xm <= 0.0) or (alpha <= 0.0):
            raise ValueError("Invalid parameter(s) for the Pareto distribution")
        try:
            offset = float(params[2])
        except IndexError:
            offset = 0.0
        return ( alpha, offset, xm, )
    if distname == "poisson":
        if params == None:
            return ( ( "MU", "expected number of occurences", ), )
        if tostd:
            return ( params[0], )
        if len(params) != 1:
            raise ValueError("One parameter expected for the Poisson distribution")
        mu = float(params[0])
        if mu <= 0.0:
            raise ValueError("Invalid parameter for the Poisson distribution")
        return ( mu, )
    if distname == "randint":
        if params == None:
            return ( ( "MIN", "minimum integer", ),
                     ( "MAX", "maximum integer (included)", ), )
        if tostd:
            return ( params[0], params[1] - 1, )
        if len(params) != 2:
            raise ValueError("Two parameters expected for the Random-Integer distribution")
        min = int(params[0])
        max = int(params[1])
        # randint takes int values, thus float values are truncated
        # this could lead to unexpected behavior (eg, one might expect
        # (0.9,10.1) to be treated as [1,11) but instead it becomes [0,10)
        minflt = float(params[0])
        maxflt = float(params[1])
        if (min >= max) or (min != minflt) or (max != maxflt):
            raise ValueError("Invalid parameters for the Random-Integer distribution")
        return ( min, max + 1, )
    if distname == "t":
        if params == None:
            return ( ( "DF", "degrees of freedom", ), )
        if tostd:
            return ( params[0], params[1], params[2], )
        if (len(params) < 1) or (len(params) > 3):
            raise ValueError("One to three parameters expected for the Students-T distribution")
        degfree = float(params[0])
        if degfree <= 0.0:
            raise ValueError("Invalid parameter for the Students-T distribution")
        offset = 0.0
        scaling = 1.0
        try:
            offset = float(params[1])
            scaling = float(params[2])
        except IndexError:
            pass
        return ( degfree, offset, scaling, )
    if distname == "uniform":
        if params == None:
            return ( ( "MIN", "minimum", ),
                     ( "MAX", "maximum", ), )
        if tostd:
            return ( params[0], params[0] + params[1], )
        if len(params) != 2:
            raise ValueError("Two parameters expected for the Uniform distribution")
        min = float(params[0])
        max = float(params[1])
        if min >= max:
            raise ValueError("Invalid parameters for the Uniform distribution")
        # these are the "loc" and "scale" parameters for the uniform distribution
        return ( min, max - min, )
    if distname == "weibull_min":
        if params == None:
            return ( ( "K", "shape", ),
                     ( "LAMBDA", "scale", ), )
        if tostd:
            return ( params[0], params[2], params[1], )
        if (len(params) < 2) or (len(params) > 3):
            raise ValueError("Two or three parameters expected for the Weibull distribution")
        k =  float(params[0])
        lambdaflt = float(params[1])
        if (k <= 0.0) or (lambdaflt <= 0.0):
            raise ValueError("Invalid parameter(s) for the Weibull distribution")
        try:
            offset = float(params[2])
        except IndexError:
            offset = 0.0
        return ( k, offset, lambdaflt, )
    return None


def getdistrib(distribname, distribparams):
    """
    Creates and returns scipy.stats "frozen" probability distribution
    object.  Converts the "standard" parameters (including ordering)
    for a distribution to appropriate parameters for calling the
    constructor of the scipy.stats frozen distribution object.

    Arguments:
       distribname - name of the probability distribution
       distribparams - tuple/list/array of standard input parameters

    Returns:
       the scipy.stats "frozen" probability distribution object
           described by distribname and distribparams

    Raises:
       ValueError if the distribution name is not recognized by this
                  routine or if the distribution parameters are invalid
    """
    if (distribname == None) or (distribparams == None):
        raise ValueError("Neither distribname nor distribparams can be None")
    distscipyname = getdistname(distribname)
    if distscipyname == None:
        raise ValueError("Unknown probability function %s" % str(distribname))
    distscipyparams = getdistparams(distscipyname, distribparams)
    if distscipyparams == None:
        raise ValueError("Unknown (for params) probability function %s" % str(distribname))
    distfunc = eval("scipy.stats.%s" % distscipyname)
    distrib = distfunc(*distscipyparams)
    return distrib


def getfitparams(values, distribname, estparams):
    """
    Returns a tuple of "standard" parameters (including ordering) for a
    continuous probability distribution type named in distribname that
    best fits the distribution of data given in values (a 1-D array of
    data with no missing values).  Initial estimates for these "standard"
    parameters are given in estparams.
    """
    if (distribname == None) or (estparams == None):
        raise ValueError("Neither distribname nor estparams can be None")
    distscipyname = getdistname(distribname)
    if distscipyname == None:
        raise ValueError("Unknown probability function %s" % str(distribname))
    estscipyparams = getdistparams(distscipyname, estparams)
    if estscipyparams == None:
        raise ValueError("Unknown (for params) probability function %s" % str(distribname))
    try:
        fitfunc = eval("scipy.stats.%s.fit" % distscipyname)
    except AttributeError:
        raise ValueError("No fit function for probability function %s" % str(distribname))
    if distscipyname == "uniform":
        # "params" keyword for the uniform distribution does not work as expected
        fitscipyparams = fitfunc(values, loc=estscipyparams[0], scale=estscipyparams[1])
    else:
        fitscipyparams = fitfunc(values, params=estscipyparams)
    return getdistparams(distscipyname, fitscipyparams, tostd=True)


def getinitdict(distribname, funcname):
    """
    Returns a dictionary appropriate for the return value of ferret_init
    in a Ferret stats_<disribname>_<funcname> PyEF

    Arguments:
       distribname - name of the probability distribution
       funcname - name of the scipy.stats probability distribution function
    """
    # generate a long function name from the scipy.stats function name
    if ( funcname == "cdf" ):
        funcaction = "calculate"
        funcreturn = "cumulative density function values"
    elif ( funcname == "isf" ):
        funcaction = "calculate"
        funcreturn = "inversion survival function values"
    elif ( funcname == "pdf" ):
        funcaction = "calculate"
        funcreturn = "probability distribution function values"
    elif ( funcname == "pmf" ):
        funcaction = "calculate"
        funcreturn = "probability mass function values"
    elif ( funcname == "ppf" ):
        funcaction = "calculate"
        funcreturn = "percent point function values"
    elif ( funcname == "sf" ):
        funcaction = "calculate"
        funcreturn = "survival function values"
    elif ( funcname == "rvs" ):
        funcaction = "assign"
        funcreturn = "random variates"
    else:
        raise ValueError("Unsupported scipy.stats function name '%s'" % funcname)
    # Get the distribution parameters information
    distscipyname = getdistname(distribname)
    paramdescripts = getdistparams(distscipyname, None)
    numargs = len(paramdescripts) + 1
    descript = "Returns array of %s for %s prob. distrib." % (funcreturn, distribname)
    axes = [ pyferret.AXIS_IMPLIED_BY_ARGS ] * pyferret.MAX_FERRET_NDIM
    argtypes = [ pyferret.FLOAT_ARRAY ] * numargs
    influences = [ [ True ] * pyferret.MAX_FERRET_NDIM ] * numargs
    if ( numargs == 2 ):
        # info for distributions with one parameter
        argnames = ( "PTS", paramdescripts[0][0], )
        argdescripts = ( "Point(s) at which to %s the %s" % (funcaction, funcreturn),
                         "Parameter(s) defining the %s" % paramdescripts[0][1], )
    elif (numargs == 3):
        # info for distributions with two parameters
        argnames = ( "PTS", paramdescripts[0][0], paramdescripts[1][0], )
        argdescripts = ( "Point(s) at which to %s the %s" % (funcaction, funcreturn),
                         "Parameter(s) defining the %s" % paramdescripts[0][1],
                         "Parameter(s) defining the %s" % paramdescripts[1][1], )
    elif (numargs == 4):
        # info for distributions with three parameters
        argnames = ( "PTS", paramdescripts[0][0], paramdescripts[1][0], paramdescripts[2][0], )
        argdescripts = ( "Point(s) at which to %s the %s" % (funcaction, funcreturn),
                         "Parameter(s) defining the %s" % paramdescripts[0][1],
                         "Parameter(s) defining the %s" % paramdescripts[1][1],
                         "Parameter(s) defining the %s" % paramdescripts[2][1], )
    else:
        raise ValueError("Unexpected number of arguments: %d" % numargs)
    # Create and return the dictionary
    return { "numargs": numargs,
             "descript": descript,
             "axes": axes,
             "argnames": argnames,
             "argdescripts": argdescripts,
             "argtypes": argtypes,
             "influences": influences, }


def getdistribfunc(distrib, funcname):
    """
    Returns the distrib.funcname function for recognized funcnames
    """
    if ( funcname == "cdf" ):
        return distrib.cdf
    elif ( funcname == "isf" ):
        return distrib.isf
    elif ( funcname == "pdf" ):
        return distrib.pdf
    elif ( funcname == "pmf" ):
        return distrib.pmf
    elif ( funcname == "ppf" ):
        return distrib.ppf
    elif ( funcname == "sf" ):
        return distrib.sf
    elif ( funcname == "rvs" ):
        return distrib.rvs
    else:
        raise ValueError("Unsupported scipy.stats function name '%s'" % funcname)


def assignresultsarray(distribname, funcname, result, resbdf, inputs, inpbdfs):
    """
    Assigns result with the funcname function values for the distribname
    probability distributions defined by parameters in inputs[1:]
    using the abscissa or template values given in inputs[0].
    """
    # get the masks for the PTS data
    pts = inputs[0]
    badmask = ( numpy.fabs(pts - inpbdfs[0]) < 1.0E-7 )
    badmask = numpy.logical_or(badmask, numpy.isnan(pts))
    goodmask = numpy.logical_not(badmask)
    # figure out the axes for the parameters
    numparams = len(inputs) - 1
    if numparams > 0:
        par1axis = -1
        for k in xrange(pyferret.MAX_FERRET_NDIM):
            if inputs[1].shape[k] > 1:
                if par1axis != -1:
                    raise ValueError("Parameters arrays can have only one defined, non-singular axis")
                par1axis = k
        if par1axis >= 0:
            if pts.shape[par1axis] > 1:
                raise ValueError("Unexpected error: shape[%d] of PTS and PAR1 both > 1" % par1axis)
        par1vals = inputs[1].reshape(-1)
        # temporary results array for a given first parameter
        tmp1result = numpy.empty(pts.shape, dtype=numpy.float64, order='F')
        tmp1result[badmask] = resbdf
    if numparams > 1:
        par2axis = -1
        for k in xrange(pyferret.MAX_FERRET_NDIM):
            if inputs[2].shape[k] > 1:
                if par2axis != -1:
                    raise ValueError("Parameters arrays can have only one defined, non-singular axis")
                par2axis = k
        if par2axis >= 0:
            if pts.shape[par2axis] > 1:
                raise ValueError("Unexpected error: shape[%d] of PTS and PAR2 both > 1" % par2axis)
            if par1axis == par2axis:
                raise ValueError("Unexpected error: shape[%d] of PAR1 and PAR2 both > 1" % par2axis)
        par2vals = inputs[2].reshape(-1)
        # temporary results array for a given second parameter and all of the first parameters
        if par1axis == -1:
            tmp2result = tmp1result
        else:
            shape = list(tmp1result.shape)
            shape[par1axis] = len(par1vals)
            tmp2result = numpy.empty(shape, dtype=numpy.float64, order='F')
    if numparams > 2:
        par3axis = -1
        for k in xrange(pyferret.MAX_FERRET_NDIM):
            if inputs[3].shape[k] > 1:
                if par3axis != -1:
                    raise ValueError("Parameters arrays can have only one defined, non-singular axis")
                par3axis = k
        if par3axis >= 0:
            if pts.shape[par3axis] > 1:
                raise ValueError("Unexpected error: shape[%d] of PTS and PAR3 both > 1" % par3axis)
            if par1axis == par3axis:
                raise ValueError("Unexpected error: shape[%d] of PAR1 and PAR3 both > 1" % par3axis)
            if par2axis == par3axis:
                raise ValueError("Unexpected error: shape[%d] of PAR2 and PAR3 both > 1" % par3axis)
        par3vals = inputs[3].reshape(-1)
        # temporary results array for a given third parameter and all of the first and second parameters
        if par2axis == -1:
            tmp3result = tmp2result
        else:
            shape = list(tmp2result.shape)
            shape[par2axis] = len(par2vals)
            tmp3result = numpy.empty(shape, dtype=numpy.float64, order='F')
    # deal with each number of parameters separately when assigning results
    if numparams == 1:
        for j in xrange(len(par1vals)):
            try:
                if math.isnan(par1vals[j]) or (math.fabs(par1vals[j] - inpbdfs[1]) < 1.0E-7):
                    raise ValueError
                distrib = getdistrib(distribname, ( par1vals[j], ))
                distribfunc = getdistribfunc(distrib, funcname)
                if funcname == "rvs":
                    tmp1result[goodmask] = getdistribfunc(distrib, funcname)(len(pts[goodmask]))
                else:
                    tmp1result[goodmask] = getdistribfunc(distrib, funcname)(pts[goodmask])
            except ValueError:
                tmp1result[goodmask] = resbdf
            # Appropriately assign the tmp1result to result
            if par1axis == -1:
                result[:, :, :, :, :, :] = tmp1result
            elif par1axis == 0:
                result[j, :, :, :, :, :] = tmp1result[0, :, :, :, :, :]
            elif par1axis == 1:
                result[:, j, :, :, :, :] = tmp1result[:, 0, :, :, :, :]
            elif par1axis == 2:
                result[:, :, j, :, :, :] = tmp1result[:, :, 0, :, :, :]
            elif par1axis == 3:
                result[:, :, :, j, :, :] = tmp1result[:, :, :, 0, :, :]
            elif par1axis == 4:
                result[:, :, :, :, j, :] = tmp1result[:, :, :, :, 0, :]
            elif par1axis == 5:
                result[:, :, :, :, :, j] = tmp1result[:, :, :, :, :, 0]
            else:
                raise ValueError("Unexpected par1axis of %d" % par1axis)
    elif numparams == 2:
        for k in xrange(len(par2vals)):
            for j in xrange(len(par1vals)):
                try:
                    if math.isnan(par2vals[k]) or (math.fabs(par2vals[k] - inpbdfs[2]) < 1.0E-7) or \
                       math.isnan(par1vals[j]) or (math.fabs(par1vals[j] - inpbdfs[1]) < 1.0E-7):
                        raise ValueError
                    distrib = getdistrib(distribname, ( par1vals[j], par2vals[k], ))
                    distribfunc = getdistribfunc(distrib, funcname)
                    if funcname == "rvs":
                        tmp1result[goodmask] = getdistribfunc(distrib, funcname)(len(pts[goodmask]))
                    else:
                        tmp1result[goodmask] = getdistribfunc(distrib, funcname)(pts[goodmask])
                except ValueError:
                    tmp1result[goodmask] = resbdf
                # Appropriately assign the tmp1result to tmp2result
                if par1axis == -1:
                    # in this case tmp2result is tmp1result
                    pass
                elif par1axis == 0:
                    tmp2result[j, :, :, :, :, :] = tmp1result[0, :, :, :, :, :]
                elif par1axis == 1:
                    tmp2result[:, j, :, :, :, :] = tmp1result[:, 0, :, :, :, :]
                elif par1axis == 2:
                    tmp2result[:, :, j, :, :, :] = tmp1result[:, :, 0, :, :, :]
                elif par1axis == 3:
                    tmp2result[:, :, :, j, :, :] = tmp1result[:, :, :, 0, :, :]
                elif par1axis == 4:
                    tmp2result[:, :, :, :, j, :] = tmp1result[:, :, :, :, 0, :]
                elif par1axis == 5:
                    tmp2result[:, :, :, :, :, j] = tmp1result[:, :, :, :, :, 0]
                else:
                    raise ValueError("Unexpected par1axis of %d" % par1axis)
            # Appropriately assign the tmp2result to result
            if par2axis == -1:
                result[:, :, :, :, :, :] = tmp2result
            elif par2axis == 0:
                result[k, :, :, :, :, :] = tmp2result[0, :, :, :, :, :]
            elif par2axis == 1:
                result[:, k, :, :, :, :] = tmp2result[:, 0, :, :, :, :]
            elif par2axis == 2:
                result[:, :, k, :, :, :] = tmp2result[:, :, 0, :, :, :]
            elif par2axis == 3:
                result[:, :, :, k, :, :] = tmp2result[:, :, :, 0, :, :]
            elif par2axis == 4:
                result[:, :, :, :, k, :] = tmp2result[:, :, :, :, 0, :]
            elif par2axis == 5:
                result[:, :, :, :, :, k] = tmp2result[:, :, :, :, :, 0]
            else:
                raise ValueError("Unexpected par2axis of %d" % par2axis)
    elif numparams == 3:
        for q in xrange(len(par3vals)):
            for k in xrange(len(par2vals)):
                for j in xrange(len(par1vals)):
                    try:
                        if math.isnan(par3vals[q]) or (math.fabs(par3vals[q] - inpbdfs[3]) < 1.0E-7) or \
                           math.isnan(par2vals[k]) or (math.fabs(par2vals[k] - inpbdfs[2]) < 1.0E-7) or \
                           math.isnan(par1vals[j]) or (math.fabs(par1vals[j] - inpbdfs[1]) < 1.0E-7):
                            raise ValueError
                        distrib = getdistrib(distribname, ( par1vals[j], par2vals[k], par3vals[q], ))
                        distribfunc = getdistribfunc(distrib, funcname)
                        if funcname == "rvs":
                            tmp1result[goodmask] = getdistribfunc(distrib, funcname)(len(pts[goodmask]))
                        else:
                            tmp1result[goodmask] = getdistribfunc(distrib, funcname)(pts[goodmask])
                    except ValueError:
                        tmp1result[goodmask] = resbdf
                    # Appropriately assign the tmp1result to tmp2result
                    if par1axis == -1:
                        # in this case tmp2result is tmp1result
                        pass
                    elif par1axis == 0:
                        tmp2result[j, :, :, :, :, :] = tmp1result[0, :, :, :, :, :]
                    elif par1axis == 1:
                        tmp2result[:, j, :, :, :, :] = tmp1result[:, 0, :, :, :, :]
                    elif par1axis == 2:
                        tmp2result[:, :, j, :, :, :] = tmp1result[:, :, 0, :, :, :]
                    elif par1axis == 3:
                        tmp2result[:, :, :, j, :, :] = tmp1result[:, :, :, 0, :, :]
                    elif par1axis == 4:
                        tmp2result[:, :, :, :, j, :] = tmp1result[:, :, :, :, 0, :]
                    elif par1axis == 5:
                        tmp2result[:, :, :, :, :, j] = tmp1result[:, :, :, :, :, 0]
                    else:
                        raise ValueError("Unexpected par1axis of %d" % par1axis)
                # Appropriately assign the tmp2result to tmp3result
                if par2axis == -1:
                    # in this case tmp3result is tmp2result
                    pass
                elif par2axis == 0:
                    tmp3result[k, :, :, :, :, :] = tmp2result[0, :, :, :, :, :]
                elif par2axis == 1:
                    tmp3result[:, k, :, :, :, :] = tmp2result[:, 0, :, :, :, :]
                elif par2axis == 2:
                    tmp3result[:, :, k, :, :, :] = tmp2result[:, :, 0, :, :, :]
                elif par2axis == 3:
                    tmp3result[:, :, :, k, :, :] = tmp2result[:, :, :, 0, :, :]
                elif par2axis == 4:
                    tmp3result[:, :, :, :, k, :] = tmp2result[:, :, :, :, 0, :]
                elif par2axis == 5:
                    tmp3result[:, :, :, :, :, k] = tmp2result[:, :, :, :, :, 0]
                else:
                    raise ValueError("Unexpected par2axis of %d" % par2axis)
            # Appropriately assign the tmp3result to result
            if par3axis == -1:
                result[:, :, :, :, :, :] = tmp3result
            elif par3axis == 0:
                result[q, :, :, :, :, :] = tmp3result[0, :, :, :, :, :]
            elif par3axis == 1:
                result[:, q, :, :, :, :] = tmp3result[:, 0, :, :, :, :]
            elif par3axis == 2:
                result[:, :, q, :, :, :] = tmp3result[:, :, 0, :, :, :]
            elif par3axis == 3:
                result[:, :, :, q, :, :] = tmp3result[:, :, :, 0, :, :]
            elif par3axis == 4:
                result[:, :, :, :, q, :] = tmp3result[:, :, :, :, 0, :]
            elif par3axis == 5:
                result[:, :, :, :, :, q] = tmp3result[:, :, :, :, :, 0]
            else:
                raise ValueError("Unexpected par3axis of %d" % par3axis)
    else:
        raise ValueError("Unexpected number of parameters: %d" % numparams)


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # Test getdistname names
    namelist = getdistname(None)
    if len(namelist) < 22:
        raise ValueError("Too few of distributions: expected at least 22; found %d" % \
                         len(distdescripts))
    for nametuple in namelist:
        for name in nametuple:
            statsname = getdistname(name)
            if statsname != nametuple[0]:
                raise ValueError("getdistname(%s): expected %s; found %s" % \
                                 (name, nametuple[0], statsname))

    # Test the distribution scipy name and parameters given to getdistrib
    # give the expected distribution.  (Primarily that the parameters
    # are interpreted and assigned correctly.)
    # Testing of the long names is performed by the stats_helper.py script.
    # Testing of assignresultsarray is done in the stats_norm_*.py scipts.

    # Beta distribution
    distname = "beta"
    descript = getdistparams(distname, None)
    if len(descript) != 2:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d" % \
                         (distname, len(descript)))
    del descript

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
        print "%s: FAIL" % distname
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))
    del alpha, beta, foundstats, expectedstats

    # append the default loc and scale to the expected params
    distparms.append(0.0)
    distparms.append(1.0)
    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    # sample = distf.rvs(25000)
    # fitparms = getfitparams(sample, distname, distparms)
    # print "%s fitparams: %s" % (distname, str(fitparms))
    # if not numpy.allclose(fitparms, distparms, rtol=0.1, atol=0.2):
    #     print "%s: FAIL" % distname
    #     raise ValueError("fitparams of %s: expected %s; found %s" % \
    #                                   (distname, str(distparms), str(fitparms)))
    del distparms, distf, newparms # , sample, fitparms

    print "%s: PASS" % distname


    # Binomial distribution
    distname = "binom"
    descript = getdistparams(distname, None)
    if len(descript) != 2:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    del descript

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
        print "%s: FAIL" % distname
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))
    del ntrials, prob, foundstats, expectedstats

    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    # no binom.fit function
    del distparms, distf, newparms

    print "%s: PASS" % distname


    # Cauchy distribution
    distname = "cauchy"
    descript = getdistparams(distname, None)
    if len(descript) != 2:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    del descript

    m = 5.0
    gamma = 2.0
    distparms = [ m, gamma ]
    distf = getdistrib(distname, distparms)
    # mean, variance, skew, kurtosis undefined; instead check some pdf values
    xvals = numpy.arange(0.0, 10.1, 0.5)
    foundpdfs = distf.pdf(xvals)
    expectedpdfs = (gamma / numpy.pi) / ((xvals - m)**2 + gamma**2)
    if not numpy.allclose(foundpdfs, expectedpdfs):
        print "%s: FAIL" % distname
        raise ValueError("pdfs(0.0:10.1:0.5) of %s(%#.1f,%#.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedpdfs), str(foundpdfs)))
    del m, gamma, xvals, foundpdfs, expectedpdfs

    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    # sample = distf.rvs(25000)
    # fitparms = getfitparams(sample, distname, distparms)
    # print "%s fitparams: %s" % (distname, str(fitparms))
    # if not numpy.allclose(fitparms, distparms, rtol=0.1, atol=0.2):
    #     print "%s: FAIL" % distname
    #     raise ValueError("fitparams of %s: expected %s; found %s" % \
    #                                   (distname, str(distparms), str(fitparms)))
    del distparms, distf, newparms # , sample, fitparms

    print "%s: PASS" % distname


    # Chi distribution
    distname = "chi"
    descript = getdistparams(distname, None)
    if len(descript) != 1:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 1; found %d:" % \
                         (distname, len(descript)))
    del descript

    degfree = 10
    distparms = [ degfree ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    mean = math.sqrt(2.0) * scipy.special.gamma(0.5 * (degfree + 1.0)) / \
                            scipy.special.gamma(0.5 * degfree)
    variance = degfree - mean**2
    stdev = math.sqrt(variance)
    skew = mean * (1.0 - 2.0 * variance) / stdev**3
    expectedstats = ( mean,
                      variance,
                      skew,
                      2.0 * (1.0 - mean * stdev * skew - variance) / variance,
                    )
    if not numpy.allclose(foundstats, expectedstats):
        print "%s: FAIL" % distname
        raise ValueError("(mean, var, skew, kurtosis) of %s(%d.0): expected %s; found %s" % \
                          (distname, distparms[0], str(expectedstats), str(foundstats)))
    del degfree, foundstats, mean, variance, stdev, skew, expectedstats

    # append the default loc and scale to the expected params
    distparms.append(0.0)
    distparms.append(1.0)
    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    # sample = distf.rvs(25000)
    # fitparms = getfitparams(sample, distname, distparms)
    # print "%s fitparams: %s" % (distname, str(fitparms))
    # if not numpy.allclose(fitparms, distparms, rtol=0.4, atol=0.4):
    #     print "%s: FAIL" % distname
    #     raise ValueError("fitparams of %s: expected %s; found %s" % \
    #                                   (distname, str(distparms), str(fitparms)))
    del distparms, distf, newparms # , sample, fitparms

    print "%s: PASS" % distname


    # Chi-squared distribution
    distname = "chi2"
    descript = getdistparams(distname, None)
    if len(descript) != 1:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 1; found %d:" % \
                         (distname, len(descript)))
    del descript

    degfreestr = "10"
    distparms = [ degfreestr ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    degfree = float(degfreestr)
    expectedstats = ( degfree,
                      2.0 * degfree,
                      math.sqrt(8.0 / degfree),
                      12.0 / degfree,
                    )
    if not numpy.allclose(foundstats, expectedstats):
        print "%s: FAIL" % distname
        raise ValueError("(mean, var, skew, kurtosis) of %s(%s): expected %s; found %s" % \
                          (distname, distparms[0], str(expectedstats), str(foundstats)))
    del degfreestr, foundstats, expectedstats

    # append the default loc and scale to the expected (numeric) params
    distparms = [ degfree, 0.0, 1.0 ]
    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    # sample = distf.rvs(25000)
    # fitparms = getfitparams(sample, distname, distparms)
    # print "%s fitparams: %s" % (distname, str(fitparms))
    # if not numpy.allclose(fitparms, distparms, rtol=0.4, atol=0.4):
    #     print "%s: FAIL" % distname
    #     raise ValueError("fitparams of %s: expected %s; found %s" % \
    #                                   (distname, str(distparms), str(fitparms)))
    del degfree, distparms, distf, newparms # , sample, fitparms

    print "%s: PASS" % distname


    # Exponential distribution
    distname = "expon"
    descript = getdistparams(distname, None)
    if len(descript) != 1:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 1; found %d:" % \
                         (distname, len(descript)))
    del descript

    lambdaflt = 11.0
    distparms = [ lambdaflt ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( 1.0 / lambdaflt, 1.0 / lambdaflt**2, 2.0, 6.0 )
    if not numpy.allclose(foundstats, expectedstats):
        print "%s: FAIL" % distname
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f): expected %s; found %s" % \
                          (distname, distparms[0], str(expectedstats), str(foundstats)))
    del lambdaflt, foundstats, expectedstats

    # append the default loc to the expected params
    distparms.append(0.0)
    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    # sample = distf.rvs(25000)
    # fitparms = getfitparams(sample, distname, distparms)
    # print "%s fitparams: %s" % (distname, str(fitparms))
    # if not numpy.allclose(fitparms, distparms, rtol=0.1, atol=0.2):
    #     print "%s: FAIL" % distname
    #     raise ValueError("fitparams of %s: expected %s; found %s" % \
    #                                   (distname, str(distparms), str(fitparms)))
    del distparms, distf, newparms # , sample, fitparms

    print "%s: PASS" % distname


    # Exponentiated Weibull distribution
    distname = "exponweib"
    descript = getdistparams(distname, None)
    if len(descript) != 3:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 3; found %d:" % \
                         (distname, len(descript)))
    del descript

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
        print "%s: FAIL" % distname
        raise ValueError("cdfs(0.0:10.1:0.5) of %s(%#.1f,%#.1f%#.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], distparms[2], str(expectedcdfs), str(foundcdfs)))
    del k, lambdaflt, alpha, xvals, foundcdfs, expectedcdfs

    # append the default loc to the expected params
    distparms.append(0.0)
    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    # sample = distf.rvs(25000)
    # fitparms = getfitparams(sample, distname, distparms)
    # print "%s fitparams: %s" % (distname, str(fitparms))
    # if not numpy.allclose(fitparms, distparms, rtol=0.1, atol=0.2):
    #     print "%s: FAIL" % distname
    #     raise ValueError("fitparams of %s: expected %s; found %s" % \
    #                                   (distname, str(distparms), str(fitparms)))
    del distparms, distf, newparms # , sample, fitparms

    print "%s: PASS" % distname


    # F distribution
    distname = "f"
    descript = getdistparams(distname, None)
    if len(descript) != 2:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    del descript

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
        print "%s: FAIL" % distname
        raise ValueError("(mean, var) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))
    # since skew and kurtosis is not coming out as expected, check some pdf values
    xvals = numpy.arange(0.5, 10.1, 0.5)
    foundpdfs = distf.pdf(xvals)
    factor = scipy.special.gamma(0.5 * (dofn + dofd)) / \
             (scipy.special.gamma(0.5 * dofn) * scipy.special.gamma(0.5 *dofd))
    factor *= math.pow(dofn, 0.5 * dofn) * math.pow(dofd, 0.5 * dofd)
    expectedpdfs = factor * numpy.power(xvals, 0.5 * dofn - 1.0) / \
                   numpy.power(dofd + dofn * xvals, 0.5 * (dofn + dofd))
    if not numpy.allclose(foundpdfs, expectedpdfs):
        print "%s: FAIL" % distname
        raise ValueError("pdfs(0.5:10.1:0.5) of %s(%#.1f,%#.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedpdfs), str(foundpdfs)))
    del dofn, dofd, foundstats, expectedstats, xvals, foundpdfs, factor, expectedpdfs

    # append the default loc and scale to the expected params
    distparms.append(0.0)
    distparms.append(1.0)
    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    # sample = distf.rvs(25000)
    # fitparms = getfitparams(sample, distname, distparms)
    # print "%s fitparams: %s" % (distname, str(fitparms))
    # if not numpy.allclose(fitparms, distparms, rtol=0.2, atol=0.4):
    #     print "%s: FAIL" % distname
    #     raise ValueError("fitparams of %s: expected %s; found %s" % \
    #                                   (distname, str(distparms), str(fitparms)))
    del distparms, distf, newparms # , sample, fitparms

    print "%s: PASS" % distname


    # Gamma distribution
    distname = "gamma"
    descript = getdistparams(distname, None)
    if len(descript) != 2:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    del descript

    alpha = 5.0
    theta = 3.0
    distparms = [ alpha, theta ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( alpha * theta, alpha * theta**2, 2.0 / math.sqrt(alpha), 6.0 / alpha )
    if not numpy.allclose(foundstats, expectedstats):
        print "%s: FAIL" % distname
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))
    del alpha, theta, foundstats, expectedstats

    # append the default loc to the expected params
    distparms.append(0.0)
    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    sample = distf.rvs(25000)
    fitparms = getfitparams(sample, distname, distparms)
    if not numpy.allclose(fitparms, distparms, rtol=0.1, atol=0.2):
        print "%s: FAIL" % distname
        raise ValueError("fitparams of %s: expected %s; found %s" % \
                                      (distname, str(distparms), str(fitparms)))
    del distparms, distf, newparms, sample, fitparms

    print "%s: PASS" % distname


    # Geometric distribution
    distname = "geom"
    descript = getdistparams(distname, None)
    if len(descript) != 1:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 1; found %d:" % \
                         (distname, len(descript)))
    del descript

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
        print "%s: FAIL" % distname
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f): expected %s; found %s" % \
                          (distname, distparms[0], str(expectedstats), str(foundstats)))
    del prob, foundstats, expectedstats

    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    # no geom.fit function
    del distparms, distf, newparms

    print "%s: PASS" % distname


    # Hypergeometric distribution
    distname = "hypergeom"
    descript = getdistparams(distname, None)
    if len(descript) != 3:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 3; found %d:" % \
                         (distname, len(descript)))
    del descript

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
        print "%s: FAIL" % distname
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], distparms[2], str(expectedstats), str(foundstats)))
    del numtotal, numgood, numdrawn, foundstats, expectedstats

    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    # no hypergeom.fit function
    del distparms, distf, newparms

    print "%s: PASS" % distname


    # Inverse-Gamma distribution
    distname = "invgamma"
    descript = getdistparams(distname, None)
    if len(descript) != 2:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    del descript

    alpha = 7.0  # must be > 4 for the kurtosis formula
    beta = 3.0
    distparms = [ alpha, beta ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( beta / (alpha - 1.0),
                      beta**2 / ((alpha - 1.0)**2 * (alpha - 2.0)),
                      4.0 * math.sqrt(alpha - 2.0) / (alpha - 3.0),
                      (30.0 * alpha - 66.0)/ ((alpha - 3.0) * (alpha - 4.0)),
                    )
    if not numpy.allclose(foundstats, expectedstats):
        print "%s: FAIL" % distname
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))
    del alpha, beta, foundstats, expectedstats

    # append the default loc to the expected params
    distparms.append(0.0)
    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    # sample = distf.rvs(25000)
    # fitparms = getfitparams(sample, distname, distparms)
    # print "%s fitparams: %s" % (distname, str(fitparms))
    # if not numpy.allclose(fitparms, distparms, rtol=0.2, atol=0.4):
    #     print "%s: FAIL" % distname
    #     raise ValueError("fitparams of %s: expected %s; found %s" % \
    #                                   (distname, str(distparms), str(fitparms)))
    del distparms, distf, newparms # , sample, fitparms

    print "%s: PASS" % distname


    # Laplace distribution
    distname = "laplace"
    descript = getdistparams(distname, None)
    if len(descript) != 2:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    del descript

    mu = 5.0
    b = 3.0
    distparms = [ mu, b ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( mu, 2.0 * b**2, 0.0, 3.0 )
    if not numpy.allclose(foundstats, expectedstats):
        print "%s: FAIL" % distname
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))
    del mu, b, foundstats, expectedstats

    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    sample = distf.rvs(25000)
    fitparms = getfitparams(sample, distname, distparms)
    if not numpy.allclose(fitparms, distparms, rtol=0.1, atol=0.2):
        print "%s: FAIL" % distname
        raise ValueError("fitparams of %s: expected %s; found %s" % \
                                      (distname, str(distparms), str(fitparms)))
    del distparms, distf, newparms, sample, fitparms

    print "%s: PASS" % distname


    # Log-normal distribution
    distname = "lognorm"
    descript = getdistparams(distname, None)
    if len(descript) != 2:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    del descript

    mu = 0.8
    sigma = 0.5
    distparms = [ mu, sigma ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( math.exp(mu + 0.5 * sigma**2),
                      math.exp(2.0 * mu + sigma**2) * (math.exp(sigma**2) - 1.0),
                      (2.0 + math.exp(sigma**2)) * math.sqrt(math.exp(sigma**2) - 1.0),
                      math.exp(4.0 * sigma**2) + 2.0 * math.exp(3.0 * sigma**2) + 3.0 * math.exp(2.0 * sigma**2) - 6,
                    )
    if not numpy.allclose(foundstats, expectedstats):
        print "%s: FAIL" % distname
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))
    del mu, sigma, foundstats, expectedstats

    # append the default loc to the expected params
    distparms.append(0.0)
    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    # sample = distf.rvs(25000)
    # fitparms = getfitparams(sample, distname, distparms)
    # print "%s fitparams: %s" % (distname, str(fitparms))
    # if not numpy.allclose(fitparms, distparms, rtol=0.1, atol=0.2):
    #     print "%s: FAIL" % distname
    #     raise ValueError("fitparams of %s: expected %s; found %s" % \
    #                                   (distname, str(distparms), str(fitparms)))
    del distparms, distf, newparms # , sample, fitparms

    print "%s: PASS" % distname


    # Negative-binomial distribution
    distname = "nbinom"
    descript = getdistparams(distname, None)
    if len(descript) != 2:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    del descript

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
        print "%s: FAIL" % distname
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))
    del numsuccess, prob, foundstats, expectedstats

    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    # no nbinom.fit function
    del distparms, distf, newparms

    print "%s: PASS" % distname


    # Normal distribution
    distname = "norm"
    descript = getdistparams(distname, None)
    if len(descript) != 2:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    del descript

    mu = 5.0
    sigma = 3.0
    distparms = numpy.array([ mu, sigma ], dtype=numpy.float64)
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( mu, sigma**2, 0.0, 0.0 )
    if not numpy.allclose(foundstats, expectedstats):
        print "%s: FAIL" % distname
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))
    del mu, sigma, foundstats, expectedstats

    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    sample = distf.rvs(25000)
    fitparms = getfitparams(sample, distname, distparms)
    if not numpy.allclose(fitparms, distparms, rtol=0.1, atol=0.2):
        print "%s: FAIL" % distname
        raise ValueError("fitparams of %s: expected %s; found %s" % \
                                      (distname, str(distparms), str(fitparms)))
    del distparms, distf, newparms, sample, fitparms

    print "%s: PASS" % distname


    # Pareto distribution
    distname = "pareto"
    descript = getdistparams(distname, None)
    if len(descript) != 2:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    del descript

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
        print "%s: FAIL" % distname
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))
    del xm, alpha, foundstats, expectedstats

    # append the default loc to the expected params
    distparms.append(0.0)
    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    sample = distf.rvs(25000)
    # fitparms = getfitparams(sample, distname, distparms)
    # print "%s fitparams: %s" % (distname, str(fitparms))
    # if not numpy.allclose(fitparms, distparms, rtol=0.1, atol=0.2):
    #     print "%s: FAIL" % distname
    #     raise ValueError("fitparams of %s: expected %s; found %s" % \
    #                                   (distname, str(distparms), str(fitparms)))
    del distparms, distf, newparms # , sample, fitparms

    print "%s: PASS" % distname


    # Poisson distribution
    distname = "poisson"
    descript = getdistparams(distname, None)
    if len(descript) != 1:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 1; found %d:" % \
                         (distname, len(descript)))
    del descript

    mu = 7.0
    distparms = [ mu ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( mu, mu, 1.0 / math.sqrt(mu), 1.0 / mu )
    if not numpy.allclose(foundstats, expectedstats):
        print "%s: FAIL" % distname
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f): expected %s; found %s" % \
                          (distname, distparms[0], str(expectedstats), str(foundstats)))
    del mu, foundstats, expectedstats

    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    # no poisson.fit function
    del distparms, distf, newparms

    print "%s: PASS" % distname


    # Random Integer (Discrete Uniform) distribution
    distname = "randint"
    descript = getdistparams(distname, None)
    if len(descript) != 2:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    del descript

    a = -5.0
    b = 13.0
    distparms = [ a, b ]
    distf = getdistrib(distname, distparms)
    # foundstats = distf.stats("mvsk")
    foundstats = distf.stats("mvs")
    n = b - a + 1.0
    # expectedstats = ( 0.5 * (a + b), (n**2 - 1.0) / 12.0, 0.0, -6.0 * (n**2 + 1) / (5.0 * (n**2 - 1)) )
    expectedstats = ( 0.5 * (a + b), (n**2 - 1.0) / 12.0, 0.0, )
    if not numpy.allclose(foundstats, expectedstats):
        print "%s: FAIL" % distname
        # raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
        raise ValueError("(mean, var, skew) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))
    xvals = numpy.arange(a - 1.0, b + 1.1, 1.0)
    expectedpmfs = numpy.ones((n+2,), dtype=float) / n
    expectedpmfs[0] = 0.0
    expectedpmfs[n+1] = 0.0
    foundpmfs = distf.pmf(xvals)
    if not numpy.allclose(foundpmfs, expectedpmfs):
        print "%s: FAIL" % distname
        raise ValueError("pmfs(%.1f:%.1f:1.0) of %s(%.1f, %.1f): expected %s; found %s" % \
              (a - 1.0, b + 1.1, distname, distparms[0], distparms[1], str(expectedpmfs), str(foundpmfs)))
    del a, b, foundstats, n, expectedstats, xvals, expectedpmfs, foundpmfs

    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    # no randint.fit function
    del distparms, distf, newparms

    print "%s: PASS" % distname


    # Student's-t distribution
    distname = "t"
    descript = getdistparams(distname, None)
    if len(descript) != 1:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 1; found %d:" % \
                         (distname, len(descript)))
    del descript

    degfree = 11.0
    distparms = [ degfree ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( 0.0, degfree / (degfree - 2.0), 0.0, 6.0 / (degfree - 4.0) )
    if not numpy.allclose(foundstats, expectedstats):
        print "%s: FAIL" % distname
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f): expected %s; found %s" % \
                          (distname, distparms[0], str(expectedstats), str(foundstats)))
    del degfree, foundstats, expectedstats

    # append the default loc and scale to the expected params
    distparms.append(0.0)
    distparms.append(1.0)
    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    sample = distf.rvs(25000)
    fitparms = getfitparams(sample, distname, distparms)
    if not numpy.allclose(fitparms, distparms, rtol=0.1, atol=0.2):
        print "%s: FAIL" % distname
        raise ValueError("fitparams of %s: expected %s; found %s" % \
                                      (distname, str(distparms), str(fitparms)))
    del distparms, distf, newparms, sample, fitparms

    print "%s: PASS" % distname


    # Uniform distribution
    distname = "uniform"
    descript = getdistparams(distname, None)
    if len(descript) != 2:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    del descript

    a = -5.0
    b = 13.0
    distparms = [ a, b ]
    distf = getdistrib(distname, distparms)
    foundstats = distf.stats("mvsk")
    expectedstats = ( 0.5 * (a + b), (b - a)**2 / 12.0, 0.0, -6.0 / 5.0 )
    if not numpy.allclose(foundstats, expectedstats):
        print "%s: FAIL" % distname
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))
    del a, b, foundstats, expectedstats

    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    sample = distf.rvs(25000)
    fitparms = getfitparams(sample, distname, distparms)
    if not numpy.allclose(fitparms, distparms, rtol=0.1, atol=0.2):
        print "%s: FAIL" % distname
        raise ValueError("fitparams of %s: expected %s; found %s" % \
                                      (distname, str(distparms), str(fitparms)))
    del distparms, distf, newparms, sample, fitparms

    print "%s: PASS" % distname


    # Weibull distribution
    distname = "weibull_min"
    descript = getdistparams(distname, None)
    if len(descript) != 2:
        print "%s: FAIL" % distname
        raise ValueError("number of parameter description pairs for %s: expected 2; found %d:" % \
                         (distname, len(descript)))
    del descript

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
        print "%s: FAIL" % distname
        raise ValueError("(mean, var, skew, kurtosis) of %s(%.1f, %.1f): expected %s; found %s" % \
                          (distname, distparms[0], distparms[1], str(expectedstats), str(foundstats)))
    del k, lambdaflt, foundstats, gam1, gam2, gam3, gam4, mu, sigma, expectedstats

    # append the default loc to the expected params
    distparms.append(0.0)
    newparms = getdistparams(distname, getdistparams(distname, distparms, tostd=False), tostd=True)
    if not numpy.allclose(newparms, distparms):
        print "%s: FAIL" % distname
        raise ValueError("conversion of full %s params to scipy then back to std changed" % distname)
    # sample = distf.rvs(25000)
    # fitparms = getfitparams(sample, distname, distparms)
    # print "%s fitparams: %s" % (distname, str(fitparms))
    # if not numpy.allclose(fitparms, distparms, rtol=0.1, atol=0.2):
    #     print "%s: FAIL" % distname
    #     raise ValueError("fitparams of %s: expected %s; found %s" % \
    #                                   (distname, str(distparms), str(fitparms)))
    del distparms, distf, newparms # , sample, fitparms

    print "%s: PASS" % distname


    # All successful
    print "Success"

