"""
Returns the array of probability distribution function values
for the Normal probability distribution
using the given arrays for the abscissa or template
values and each of the parameters values.
"""
import numpy
import pyferret
import pyferret.stats
import scipy.stats

DISTRIB_NAME = "Normal"
FUNC_NAME = "pdf"


def ferret_init(id):
    """
    Initialization for the stats_norm_pdf Ferret PyEF
    """
    return pyferret.stats.getinitdict(DISTRIB_NAME, FUNC_NAME)


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Result array assignment for the stats_norm_pdf Ferret PyEF
    """
    pyferret.stats.assignresultsarray(DISTRIB_NAME, FUNC_NAME,
                                      result, resbdf, inputs, inpbdfs)


#
# The rest of this is just for testing this module at the command line
#
if __name__ == "__main__":
    # make sure ferret_init does not have problems
    info = ferret_init(0)

    # Normal distribution in the YZ plane, MUs on the X axis, SIGMAs on the T
    xdim = 6
    ydim = 25
    zdim = 15
    tdim = 4
    yzvals = numpy.linspace(0.0, 100.0, ydim * zdim)
    mus = numpy.linspace(20.0, 45.0, xdim)
    sigmas = numpy.linspace(8.0, 14.0, tdim)
    pdfs = numpy.empty((xdim, ydim, zdim, tdim, 1, 1), dtype=numpy.float64, order='F')
    for i in xrange(xdim):
        for q in xrange(tdim):
            distf = scipy.stats.norm(mus[i], sigmas[q])
            values = distf.pdf(yzvals)
            pdfs[i, :, :, q, 0, 0] = values.reshape((ydim, zdim), order='F')
    # configure arrays for ferret_compute
    yzvals = numpy.array(yzvals, dtype=numpy.float64).reshape((1, ydim, zdim, 1, 1, 1), order='F')
    mus = numpy.array(mus, dtype=numpy.float64).reshape((xdim, 1, 1, 1, 1, 1), order='F')
    sigmas = numpy.array(sigmas, dtype=numpy.float64).reshape((1, 1, 1 , tdim, 1, 1), order='F')
    inpbdfs = numpy.array([-9999.0, -8888.0, -7777.0], dtype=numpy.float64)
    resbdf = numpy.array([-6666.0], dtype=numpy.float64)
    # Throw in some undefined values
    index = 0
    for k in xrange(zdim):
        for j in xrange(ydim):
            if (index % 13) == 3:
                abscissa[0, j, k, 0, 0, 0] = inpbdfs[0]
                pdfs[:, j, k, :, 0, 0] = resbdf[0]
    mus[4, 0, 0, 0, 0, 0] = inpbdfs[1]
    pdfs[4, :, :, :, 0, 0] = resbdf[0]
    sigmas[0, 0, 0, 1, 0, 0] = inpbdfs[2]
    pdfs[:, :, :, 1, 0, 0] = resbdf[0]
    # Get the result from ferret_compute and compare
    result = -5555.0 * numpy.ones((xdim, ydim, zdim, tdim), dtype=numpy.float64, order='F')
    ferret_compute(0, result, resbdf, (yzvals, mus, sigmas), inpbdfs)
    print "Expect =\n%s" % str(pdfs)
    if not numpy.allclose(result, pdfs):
        print "Expect (flattened) =\n%s" % str(pdfs.reshape(-1))
        print "Result (flattened) =\n%s" % str(result.reshape(-1))
        raise ValueError("Unexpected result")

    # All successful
    print "Success"

