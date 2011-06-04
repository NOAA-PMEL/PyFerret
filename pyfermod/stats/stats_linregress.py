"""
Returns parameters resulting from a linear regression of one set
of given data against another set of given data.
"""
import math
import numpy
import pyferret
import scipy.stats


def ferret_init(id):
    """
    Initialization for the stats_linregress python-backed ferret external function
    """
    retdict = { "numargs": 2,
                "descript": "Returns slope, intercept, correlation coeff (r), and num good pts for a linear regression",
                "axes": ( pyferret.AXIS_CUSTOM,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST, ),
                "argnames": ( "XVALS", "YVALS", ),
                "argdescripts": ( "Abscissa values for the linear regression fit",
                                  "Ordinate values for the linear regression fit", ),
                "argtypes": ( pyferret.FLOAT_ARG, pyferret.FLOAT_ARG, ),
                "influences": ( ( False, False, False, False, ),
                                ( False, False, False, False, ), ),
              }
    return retdict


def ferret_custom_axes(id):
    """
    Define the limits and (unit)name of the custom axis of the
    array containing the returned parameters.  The values returned
    are from the scipy.stats.linregress function: slope, intercept,
    and correlation coefficient; plus the number of defined points
    used in the fitting.  The probability of zero slope and
    standard error of the estimate computed by linregress are not
    returned.  (The standard error of the estimate returned is
    incorrect.  The value returned is close but larger than the
    square of the correct value.)
    """
    return ( ( 1, 4, 1, "PARAMS", False, ), None, None, None, )


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Assigns result with parameters for the linear regression of the
    ordinate values given inputs[1] to the abscissa values given in
    inputs[0].  If the input arrays inputs[0] and inputs[1] do not
    have the same shape (the same lengths of each of the four axes),
    then then must each have only one defined non-singular axis with
    matching lengths.  Result is assigned the slope, intercept, and
    sample correlation coefficient from scipy.stats.linregress as
    well as the number of defined points used in the fitting.  The
    probability of zero slope and standard error of the estimate
    computed by linregress are not returned.  (The standard error of
    the estimate returned is incorrect.  The value returned is close
    but larger than the square of the correct value.)
    """
    if inputs[0].shape != inputs[1].shape :
        errmsg = "XVALS and YVALS must either have identical dimensions or "\
            "both have only one defined non-singular axis of the same length"
        lena = 1
        leno = 1
        for k in xrange(4):
            if inputs[0].shape[k] > 1:
                if lena != 1:
                    raise ValueError(errmsg)
                lena = inputs[0].shape[k]
        for k in xrange(4):
            if inputs[1].shape[k] > 1:
                if leno != 1:
                    raise ValueError(errmsg)
                leno = inputs[1].shape[k]
        if lena != leno:
            raise ValueError(errmsg)
    abscissa = inputs[0].reshape(-1)
    ordinate = inputs[1].reshape(-1)
    badmaska = ( numpy.fabs(abscissa - inpbdfs[0]) < 1.0E-5 )
    badmaska = numpy.logical_or(badmaska, numpy.isnan(abscissa))
    badmasko = ( numpy.fabs(ordinate - inpbdfs[1]) < 1.0E-5 )
    badmasko = numpy.logical_or(badmasko, numpy.isnan(ordinate))
    goodmask = numpy.logical_not(numpy.logical_or(badmaska, badmasko))
    # must use double precision arrays for accuracy
    xvals = numpy.array(abscissa[goodmask], dtype=numpy.float64)
    numpts = len(xvals)
    if numpts < 2:
        raise ValueError("Not enough defined points in common in XVALS and YVALS")
    yvals = numpy.array(ordinate[goodmask], dtype=numpy.float64)
    fitparams = scipy.stats.linregress(xvals, yvals)
    result[:] = resbdf
    # slope
    result[0] = fitparams[0]
    # intercept
    result[1] = fitparams[1]
    # correlation coefficient
    result[2] = fitparams[2]
    # ignore the probability of zero coefficient (fitparams[3]) and
    # the incorrect standard error of the estimate (fitparams[4])
    # number of good pts
    result[3] = numpts

#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # just make sure these calls don't throw errors
    info = ferret_init(0)
    info = ferret_custom_axes(0)
    # create some data to fit
    slope = -5.0
    intercept = 15.0
    xvals = numpy.arange(0.0, 10.0, 0.01)
    fuzz = scipy.stats.norm(0.0, 0.01).rvs(1000)
    yvals = slope * xvals + intercept + fuzz
    inpbdfs = numpy.array([-9999.0, -8888.0], dtype=numpy.float32)
    resbdf = numpy.array([-7777.0], dtype=numpy.float32)
    abscissa = numpy.empty((1, 1000, 1, 1), dtype=numpy.float32, order='F')
    ordinate = numpy.empty((1, 1, 1000, 1), dtype=numpy.float32, order='F')
    goodvals = numpy.empty((1000,), dtype=bool)
    index = 0
    numgood = 0
    for j in xrange(1000):
        if (index % 53) == 13:
            abscissa[0, j, 0, 0] = inpbdfs[0]
        else:
            abscissa[0, j, 0, 0] = xvals[j]
        if (index % 73) == 13:
            ordinate[0, 0, j, 0] = inpbdfs[1]
        else:
            ordinate[0, 0, j, 0] = yvals[j]
        if ((index % 53) == 13) or ((index % 73) == 13):
            goodvals[index] = False
        else:
            goodvals[index] = True
            numgood += 1
        index += 1
    result = -5555.0 * numpy.ones((4,), dtype=numpy.float32)
    ferret_compute(0, result, resbdf, ( abscissa, ordinate, ), inpbdfs)
    xvals = xvals[goodvals]
    xave = xvals.mean()
    yvals = yvals[goodvals]
    yave = yvals.mean()
    m = (xvals * yvals - xave * yave).sum() / (xvals * xvals - xave * xave).sum()
    b = yave - m * xave
    yxdeltas = yvals - (m * xvals + b)
    yxsumsq = (yxdeltas * yxdeltas).sum()
    # std_err_est = math.sqrt(yxsumsq / float(numgood - 2))
    ydeltas = yvals - yave
    ysumsq = (ydeltas * ydeltas).sum()
    rsq = 1.0 - (yxsumsq / ysumsq)
    r = math.sqrt(rsq)
    if m < 0.0:
        r *= -1.0
    expected = numpy.array([m, b, r, numgood], dtype=numpy.float32)
    if not numpy.allclose(result, expected, rtol=1.0E-5, atol=1.0E-5):
        raise ValueError("Linear regression fail\nexpected params:\n%s\nfound params:\n%s" % \
                         (str(expected), str(result)))

    # All successful
    print "Success"

