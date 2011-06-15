"""
Returns parameter values for a specified probability distribution type
that best describe the distribution of a given array of values.
"""
import math
import numpy
import pyferret
import pyferret.stats
import scipy.stats


def ferret_init(id):
    """
    Initialization for the stats_fit python-backed ferret external function
    """
    retdict = { "numargs": 3,
                "descript": "Returns parameters for a probability distribution that best fit all defined data values",
                "axes": ( pyferret.AXIS_CUSTOM,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST, ),
                "argnames": ( "VALS", "PDNAME", "PDPARAMS", ),
                "argdescripts": ( "Values to fit with the probability distribution",
                                  "Name of the probability distribution type to use",
                                  "Initial parameter estimates for this probability distribution", ),
                "argtypes": ( pyferret.FLOAT_ARRAY, pyferret.STRING_ONEVAL, pyferret.FLOAT_ARRAY, ),
                "influences": ( ( False, False, False, False, ),
                                ( False, False, False, False, ),
                                ( False, False, False, False, ), ),
              }
    return retdict


def ferret_custom_axes(id):
    """
    Define the limits and (unit)name of the custom axis of the array
    containing the returned parameters.  A "location" and a "scale"
    parameter, if not considered one of the "standard" parameters, is
    appended to the "standard" parameters.
    """
    return ( ( 1, 5, 1, "PDPARAMS", False, ), None, None, None, )


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Assigns result with parameters for the probability distribution type
    indicated by inputs[1] (a string) that best fit the distribution of
    values given in inputs[0].  Parameter estimates given in inputs[2]
    will be used to initialize the fitting method.  Undefined values will
    be eliminated before the fit is attempted.
    """
    distribname = inputs[1]
    estparams = inputs[2].reshape(-1)
    badmask = ( numpy.fabs(inputs[0] - inpbdfs[0]) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(inputs[0]))
    goodmask = numpy.logical_not(badmask)
    values = inputs[0][goodmask]
    # values is a flattened array
    fitparams = pyferret.stats.getfitparams(values, distribname, estparams)
    result[:] = resbdf
    if fitparams != None:
        for k in xrange(len(fitparams)):
            result[k] = fitparams[k]

#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # make sure ferret_init does not have problems
    info = ferret_init(0)

    # Normal distribution in the YZ plane
    ydimen = 100
    zdimen = 125
    mu = 5.0
    sigma = 2.0
    distf = scipy.stats.norm(mu, sigma)
    sample = distf.rvs(ydimen * zdimen)

    pfname = "norm"
    pfparams = numpy.array([mu, sigma], dtype=numpy.float32)
    inpbdfs = numpy.array([-9999.0, -8888.0, -7777.0], dtype=numpy.float32)
    resbdf = numpy.array([-6666.0], dtype=numpy.float32)
    values = numpy.empty((1, ydimen, zdimen, 1), dtype=numpy.float32, order='F')
    index = 0
    for j in xrange(ydimen):
        for k in xrange(zdimen):
            if (index % 103) == 13:
                values[0, j, k, 0] = inpbdfs[0]
            else:
                values[0, j, k, 0] = sample[index]
            index += 1
    result = -5555.0 * numpy.ones((5,), dtype=numpy.float32, order='F')
    ferret_compute(0, result, resbdf, (values, pfname, pfparams), inpbdfs)
    if (abs(result[0] - mu) > 0.2) or \
       (abs(result[1] - sigma) > 0.2) or \
       (abs(result[2] - resbdf[0]) > 1.0E-5) or \
       (abs(result[3] - resbdf[0]) > 1.0E-5) or \
       (abs(result[4] - resbdf[0]) > 1.0E-5):
        expected = ( mu, sigma, resbdf[0], resbdf[0], resbdf[0], )
        raise ValueError("Norm fit fail; expected params: %s; found %s" % (str(expected), str(result)))

    # All successful
    print "Success"

