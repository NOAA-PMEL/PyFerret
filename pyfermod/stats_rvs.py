"""
Returns the array of random variates for a probability distribution
assigned to positions corresponding to defined values in an input array.
"""
import sys
import numpy
import scipy.stats
import pyferret
import pyferret.stats


def ferret_init(id):
    """
    Initialization for the stats_rvs python-backed ferret external function
    """
    retdict = { "numargs": 3,
                "descript": "Returns random variates for a probability distribution",
                "axes": (pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS,
                         pyferret.AXIS_IMPLIED_BY_ARGS),
                "argnames": ("TEMPLATE", "PDNAME", "PDPARAMS"),
                "argdescripts": ("Template array for the random variates array to be created",
                                 "Name of a probability distribution",
                                 "Parameters for this probability distribution"),
                "argtypes": (pyferret.FLOAT_ARG, pyferret.STRING_ARG, pyferret.FLOAT_ARG),
                "influences": ((True,  True,  True,  True),
                               (False, False, False, False),
                               (False, False, False, False)),
              }
    return retdict


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Assigns result with random variates of the probability distribution 
    indicated by inputs[1] (a string) using the parameters given in inputs[2].
    Random variates will be assigned to positions corresponding to defined
    positions in input[0].
    """
    distribname = inputs[1]
    distribparams = inputs[2].reshape(-1)
    distrib = pyferret.stats.getdistrib(distribname, distribparams)
    pyferret.stats.assignrvs(result, resbdf, distrib, inputs[0], inpbdfs[0])


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # Normal distribution along the Y axis
    pfname = "norm"
    pfparams = numpy.array([5.0, 0.5], dtype=numpy.float32)
    distf = scipy.stats.norm(5.0, 0.5)
    # Get a large sample so the mean and stdev will be very close
    template = numpy.empty((19, 21, 17, 23), dtype=numpy.float32, order='F')
    expectedgood = numpy.empty((19, 21, 17, 23), dtype=bool, order='F')
    index = 0
    for i in xrange(19):
        for j in xrange(21):
            for k in xrange(17):
                for l in xrange(23):
                    index += 1
                    if (index % 53) == 0:
                        template[i, j, k, l] = -1.0
                        expectedgood[i, j, k, l] = False
                    else:
                        template[i, j, k, l] = 1.0
                        expectedgood[i, j, k, l] = True
    inpbdfs = numpy.array([-1.0, 0.0, 0.0], dtype=numpy.float32)

    result = -8888.0 * numpy.ones((19, 21, 17, 23), dtype=numpy.float32, order='F')
    resbdf = numpy.array([-65536.0], dtype=numpy.float32)

    ferret_compute(0, result, resbdf, (template, pfname, pfparams), inpbdfs)
    # probability of randomly choosing -65536.0 from this normal distribution is essentially zero
    resultgood = ( result != resbdf )
    if numpy.any( resultgood !=  expectedgood ):
        raise ValueError("Assigned random variates does not match template")
    mean = numpy.mean(result[resultgood])
    if abs(mean - 5.0) > 5.0E-3:
        raise ValueError("Mean of random sample: expected: 5.0; found: %f" % mean)
    stdev = numpy.std(result[resultgood])
    if abs(stdev - 0.5) > 5.0E-3:
        raise ValueError("Standard deviation of random sample: expected: 0.5; found: %f" % stdev)

    # All successful
    print "Success"

