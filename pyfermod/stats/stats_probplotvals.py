"""
Returns an array containing the order statistic medians for a given 
probability distribution along the X axis in the first Y axis value, 
and ordered response data of a given sample along the X axis in the
second Y axis value.  The order statistic medians are the percent 
probability function values of the probability distribution at 
regularily-spaced intervals.  The ordered response data is the sorted 
sample values.  If the sample comes from a probability distribution 
of the type given, the plot of the ordered response data against the 
order statistic medians should give a straight line whose slope is 
the offset from, and whose intercept is the scaling of, the given 
distribution.  Thus, the slope, intercept, and correlation coefficient
(r) of this fitted line are returned and the first three X elements
of the third Y axis value.
"""
import numpy
import scipy.stats
import pyferret
import pyferret.stats


def ferret_init(id):
    """
    Initialization for the stats_probplotvals Ferret PyEF
    """
    retdict = { "numargs": 3,
                "descript": "Returns [j=1] order statistic medians, " \
                                    "[j=2] ordered response data, and " \
                                    "[j=3] slope, intercept, and corr. coeff. of fitted line",
                "axes": (pyferret.AXIS_CUSTOM,
                         pyferret.AXIS_CUSTOM,
                         pyferret.AXIS_DOES_NOT_EXIST,
                         pyferret.AXIS_DOES_NOT_EXIST,
                         pyferret.AXIS_DOES_NOT_EXIST,
                         pyferret.AXIS_DOES_NOT_EXIST),
                "argnames": ("SAMPLE", "PDNAME", "PDPARAMS"),
                "argdescripts": ("Sample values for the ordered response data",
                                 "Name of a continuous probability distribution for the order statistic medians",
                                 "Parameters for this continuous probability distribution"),
                "argtypes": (pyferret.FLOAT_ARRAY, pyferret.STRING_ONEVAL, pyferret.FLOAT_ARRAY),
                "influences": ((False, False, False, False, False, False),
                               (False, False, False, False, False, False),
                               (False, False, False, False, False, False)),
              }
    return retdict


def ferret_custom_axes(id):
    """
    Custom axis information for stats_probplot_values Ferret PyEF
    """
    size = 1
    for axis in ( pyferret.X_AXIS, pyferret.Y_AXIS, pyferret.Z_AXIS, 
                  pyferret.T_AXIS, pyferret.E_AXIS, pyferret.F_AXIS ):
        axis_info = pyferret.get_axis_info(id, pyferret.ARG1, axis)
        # Note: axes normal to the data have size = -1
        num = axis_info.get("size", -1)
        if num > 1:
            size *= num
    return ( (1, size, 1, "VALUE_NUM", False, ), (1, 3, 1, "OSM,ORD,P", False, ), None, None, None, None, )


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Assigns to result[:,0,0,0,0,0] the order statistic medians 
    for the probability distribution named in inputs[1] with 
    parameters given in inputs[2].  Assigns to result[:,1,0,0,0,0] 
    the ordered response data of the sample values given in 
    inputs[0].  Assigns to result[:3,2,0,0,0,0] the slope, intercept, 
    and correlation coefficient of the line fitted to a plot of 
    result[:,1,0,0,0,0] against result[:,0,0,0,0,0].  Undefined values 
    in inputs[0] are removed at the beginning of this computation.
    """
    distribname = inputs[1]
    distname = pyferret.stats.getdistname(distribname)
    if distname == None:
        raise ValueError("Unknown probability function %s" % distribname)
    distribparams = inputs[2].reshape(-1)
    distparams = pyferret.stats.getdistparams(distname, distribparams)
    if distparams == None:
        raise ValueError("Unknown (for params) probability function %s" % distribname)

    sample = inputs[0].reshape(-1)
    expshape = ( len(sample), 3, 1, 1, 1, 1 )
    if result.shape != expshape:
        raise ValueError("Unexpected shape of results array: expected %s; found %s" % \
                         (str(expshape), str(result.shape)))
    badmask = ( numpy.fabs(sample - inpbdfs[0]) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(sample))
    goodmask = numpy.logical_not(badmask)
    ppdata = scipy.stats.probplot(sample[goodmask], distparams, distname, fit=1)
    result[badmask,0,0,0,0,0] = resbdf
    result[goodmask,0,0,0,0,0] = ppdata[0][0]
    result[badmask,1,0,0,0,0] = resbdf
    result[goodmask,1,0,0,0,0] = ppdata[0][1]
    result[3:,2,0,0,0,0] = resbdf
    result[:3,2,0,0,0,0] = ppdata[1]


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # make sure ferret_init does not have problems
    info = ferret_init(0)

    # Sample from a normal distribution
    ydim = 23
    zdim = 13
    inpundefval = -1.0E+10
    outundefval = -2.0E+10

    # select a random sample from a normal distribution
    size = ydim * zdim
    sample = scipy.stats.norm(5.0, 2.0).rvs(size)
    ordata = numpy.sort(sample)

    # compare to the standard normal distribution (mu = 0.0, sigma = 1.0)
    uvals = numpy.empty(size)
    uvals[-1] = numpy.power(0.5, 1.0 / size)
    uvals[0] = 1.0 - uvals[-1]
    uvals[1:-1] = (numpy.arange(2.0, size-0.5, 1.0) - 0.3175) / (size + 0.365)
    osmeds = scipy.stats.norm(0.0, 1.0).ppf(uvals)

    # set up for a call to ferret_compute
    pfname = "norm"
    pfparams = numpy.array([0.0, 1.0], dtype=numpy.float64)
    inpbdfs = numpy.array([inpundefval, 0.0, 0.0], dtype=numpy.float64)
    resbdf = numpy.array([outundefval], dtype=numpy.float64)
    inputarr = numpy.empty((1, ydim + 1, zdim + 1, 1, 1, 1), dtype=numpy.float64, order='F')
    expected = numpy.empty(((ydim + 1) * (zdim + 1), 3, 1, 1, 1, 1), dtype=numpy.float64, order='F')
    n = 0
    index = 0
    for j in xrange(ydim + 1):
        for k in xrange(zdim + 1):
            if (k == j) or (k == j+1) or (n >= size):
                inputarr[0, j, k, 0, 0, 0] = inpbdfs[0]
                expected[index, 0, 0, 0, 0, 0] = resbdf[0]
                expected[index, 1, 0, 0, 0, 0] = resbdf[0]
            else:
                inputarr[0, j, k, 0, 0, 0] = sample[n]
                expected[index, 0, 0, 0, 0, 0] = osmeds[n]
                expected[index, 1, 0, 0, 0, 0] = ordata[n]
                n += 1
            index += 1
    if n < size:
        raise ValueError("Unexpected result: not all values assigned")
    fitdata = scipy.stats.linregress(osmeds, ordata)
    expected[:3,2,0,0,0,0] = fitdata[:3]
    expected[3:,2,0,0,0,0] = resbdf[0]
    result = -888888.0 * numpy.ones(((ydim + 1) * (zdim + 1), 3, 1, 1, 1, 1), dtype=numpy.float64, order='F')

    # call ferret_compute and check the results
    ferret_compute(0, result, resbdf, (inputarr, pfname, pfparams), inpbdfs)
    if not numpy.allclose(result, expected):
        if not numpy.allclose(result[:,0,0,0,0,0], expected[:,0,0,0,0,0]):
            print "Expected[:,0,0,0,0,0] =\n%s" % str(expected[:,0,0,0,0,0])
            print "Result[:,0,0,0,0,0] =\n%s" % str(result[:,0,0,0,0,0])
        if not numpy.allclose(result[:,1,0,0,0,0], expected[:,1,0,0,0,0]):
            print "Expected[:,1,0,0,0,0] =\n%s" % str(expected[:,1,0,0,0,0])
            print "Result[:,1,0,0,0,0] =\n%s" % str(result[:,1,0,0,0,0])
        if not numpy.allclose(result[:3,2,0,0,0,0], expected[:3,2,0,0,0,0]):
            print "Expected[:3,2,0,0,0,0] =\n%s" % str(expected[:3,2,0,0,0,0])
            print "Result[:3,2,0,0,0,0] =\n%s" % str(result[:3,2,0,0,0,0])
        raise ValueError("Unexpected result")

    # All successful
    print "Success"
