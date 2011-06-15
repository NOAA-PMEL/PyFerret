"""
Performs a two-sided Kolmogorov-Smirnov test that the provided
sample comes from the given probability distribution function.
"""
import numpy
import pyferret
import pyferret.stats
import scipy.stats


def ferret_init(id):
    """
    Initialization for the stats_kstest1 PyEF
    """
    retdict = { "numargs": 3,
                "descript": "Returns two-sided Kolmogorov-Smirnov test stat. and prob. " \
                            "that sample comes from a pop. with given prob. distrib.",
                "axes": ( pyferret.AXIS_CUSTOM,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST, ),
                "argnames": ( "SAMPLE", "PDNAME", "PDPARAMS", ),
                "argdescripts": ( "Sample data array",
                                  "Name of a continuous probability distribution",
                                  "Parameters for this continuous probability distribution"),
                "argtypes": ( pyferret.FLOAT_ARRAY, pyferret.STRING_ONEVAL, pyferret.FLOAT_ARRAY, ),
                "influences": ( (False, False, False, False),
                                (False, False, False, False),
                                (False, False, False, False), ),
              }
    return retdict


def ferret_custom_axes(id):
    """
    Define custom axis of the stats_kstest1 Ferret PyEF
    """
    return ( ( 1, 2, 1, "KS,P", False ), None, None, None, )


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Performs a two-sided Kolmogorov-Smirnov test that the provided sample
    comes from a population with the given probability distribution function.
    The sample is given in inputs[0], the probability distribution function
    name is given in inputs[1] (a string), and the "standard" parameters for
    this probability distribution function are given in inputs[2].  The test
    statistic value and two-tailed probability are returned in result.
    Undefined data given in inputs[0] are removed before performing the test.
    """
    # get the scipy.stats distribution name from the given distribution name
    if inputs[1] == None:
        raise ValueError("The name of a probability distribution function not given")
    distscipyname = pyferret.stats.getdistname(inputs[1])
    if distscipyname == None:
        raise ValueError("Unknown or unsupported probability distribution function %s" % inputs[1])
    # get the scipy.stats distribution parameters from the given "standard" parameters
    if inputs[2] == None:
        raise ValueError("Paramaters for the probability distribution function not given")
    distscipyparams = pyferret.stats.getdistparams(distscipyname, inputs[2].reshape(-1))
    if distscipyparams == None:
        raise ValueError("Unknown or unsupported (for params) probability distribution function %s" % inputs[1])
    # get the valid sample values
    badmask = ( numpy.fabs(inputs[0] - inpbdfs[0]) < 1.0E-5 )
    badmask = numpy.logical_or(badmask, numpy.isnan(inputs[0]))
    goodmask = numpy.logical_not(badmask)
    values = inputs[0][goodmask]
    # perform the test and assign the results
    fitparams = scipy.stats.kstest(values, distscipyname, distscipyparams)
    result[:, :, :, :] = resbdf
    # Kolmogorov-Smirnov test statistic
    result[0, 0, 0, 0] = fitparams[0]
    # probability
    result[1, 0, 0, 0] = fitparams[1]


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # make sure ferret_init and ferret_custom_axes do not have problems
    info = ferret_init(0)
    info = ferret_custom_axes(0)

    # Set the seed to reproduce a problematic distribution
    # import numpy.random
    # numpy.random.seed(3333333)

    # Get a random sample from the compared distribution and from another distribution
    ydim = 200
    zdim = 150
    mu = 5.0
    sigma = 0.5
    rvsc = scipy.stats.norm(mu, sigma).rvs(ydim * zdim)
    rvsu = scipy.stats.uniform(loc=(mu + 3.0 * sigma), scale=(3.0 * sigma)).rvs(ydim * zdim)

    # setup for the call to ferret_compute
    distname = "norm"
    distparams = numpy.array([mu, sigma], dtype=numpy.float32)
    inpbdfs = numpy.array([-9999.0, -1.0, -2.0], dtype=numpy.float32)
    resbdf  = numpy.array([-8888.0], dtype=numpy.float32)
    sampc = numpy.empty((1, ydim, zdim, 1), dtype=numpy.float32, order='F')
    sampu = numpy.empty((1, ydim, zdim, 1), dtype=numpy.float32, order='F')
    index = 0
    for j in xrange(ydim):
        for k in xrange(zdim):
            if (index % 71) == 3:
                sampc[0, j, k, 0] = inpbdfs[0]
                sampu[0, j, k, 0] = inpbdfs[0]
            else:
                sampc[0, j, k, 0] = rvsc[index]
                sampu[0, j, k, 0] = rvsu[index]
            index += 1
    resultc = -7777.0 * numpy.ones((2, 1, 1, 1), dtype=numpy.float32, order='F')
    resultu = -7777.0 * numpy.ones((2, 1, 1, 1), dtype=numpy.float32, order='F')

    # call ferret_compute with data from the distribution and check the results
    ferret_compute(0, resultc, resbdf, (sampc, distname, distparams), inpbdfs)
    resultc = resultc.reshape(-1)
    print "from same dist result: %s" % str(resultc)
    if (resultc[0] < 0.00) or (resultc[0] > 0.01) or \
       (resultc[1] < 0.10) or (resultc[1] > 1.00):
        raise ValueError("Unexpected result")

    # call ferret_compute with data from a different distribution and check the results
    ferret_compute(0, resultu, resbdf, (sampu, distname, distparams), inpbdfs)
    resultu = resultu.reshape(-1)
    print "from diff dist result:  %s" % str(resultu)
    if (resultu[0] < 0.99) or (resultu[0] > 1.00) or \
       (resultu[1] < 0.00) or (resultu[1] > 0.01):
        raise ValueError("Unexpected result")

    # All successful
    print "Success"

