'''
PyFerret external function providing data partitioned into pieces 
along the ensemble axis.  Each partition is the data explained by 
a single Empirical Orthogonal Function (EOF) and its corresponding 
Time Amplitude Funtion (TAF)

@author: Karl Smith
'''

import numpy
import pyferret
import pyferret.eofanal as eofanal

def ferret_init(efid):
    '''
    Initializes the eofdatapiece function. 
    '''
    init_dict = { }
    init_dict["numargs"] = 2
    init_dict["descript"] = "Partitions data into EOF * TAF pieces " + \
        "(parts of data explained) along the ensemble axis"
    init_dict["argnames"] = ("Data",
                             "MinSignif")
    init_dict["argdescripts"] = (
        "Time-location data; defined on T and one or more of X, Y, Z",
        "Minimum fraction-of-data-explained considered significant")
    init_dict["argtypes"] = (pyferret.FLOAT_ARRAY,
                             pyferret.FLOAT_ONEVAL)
    # X, Y, Z, and T match input; E axis added as an abstract axis
    axes = [ pyferret.AXIS_IMPLIED_BY_ARGS ] * pyferret.MAX_FERRET_NDIM
    axes[pyferret.E_AXIS] = pyferret.AXIS_ABSTRACT
    axes[pyferret.F_AXIS] = pyferret.AXIS_DOES_NOT_EXIST;
    init_dict["axes"] = axes
    part_influence = [ True ] * pyferret.MAX_FERRET_NDIM
    part_influence[pyferret.E_AXIS] = False
    part_influence[pyferret.F_AXIS] = False
    no_influence = [ False ] * pyferret.MAX_FERRET_NDIM
    init_dict["influences"] = (part_influence,
                               no_influence)
    init_dict["piecemeal"] =  [ False ] * pyferret.MAX_FERRET_NDIM

    return init_dict


def ferret_result_limits(efid):
    '''
    Provides the bounds of the E abstract axis.
    The maximum number of EOFs is the number of locations. 
    '''
    maxpts = 1
    for axis in (pyferret.X_AXIS, pyferret.Y_AXIS, pyferret.Z_AXIS):
        axis_info = pyferret.get_axis_info(efid, pyferret.ARG1, axis);
        if axis_info:
            npts = axis_info.get("size", -1)
            if npts > 0:
                maxpts *= npts
    result_limits = [ None ] * pyferret.MAX_FERRET_NDIM
    result_limits[pyferret.E_AXIS] = (0, maxpts)
    return result_limits


def ferret_compute(efid, result, result_bdf, inputs, input_bdfs):
    '''
    Assign result with EOF * TAF (piece of data explained) up to 
    the number of significant EOFs.  The X,Y,Z,T data is given in
    inputs[0], the minimum fraction-of-data-explained considered
    significant is given as a single value in inputs[1].
    '''
    # verify no ensemble or forecast axis on the input data
    if inputs[pyferret.ARG1].shape[pyferret.E_AXIS] > 1:
        raise ValueError("Input data cannot have an ensemble axis")
    if inputs[pyferret.ARG1].shape[pyferret.F_AXIS] > 1:
        raise ValueError("Input data cannot have a forecast axis")
    # number of time steps in the input data
    ntime = inputs[pyferret.ARG1].shape[pyferret.T_AXIS]
    if ntime < 2:
        raise ValueError("Input data time axis does not exist or is a singleton")
    # Verify the second value is reasonable
    min_signif = float(inputs[1])
    if (min_signif < 1.0E-6) or (min_signif > (1.0 - 1.0E-6)):
        raise ValueError("MinSignif must be in [0.000001, 0.999999]")
    # Get the mask of where the data is defined
    defined_data = ( numpy.fabs(inputs[pyferret.ARG1] - 
                                input_bdfs[pyferret.ARG1]) >= 1.0E-5 )
    # Get the mask of where the data is defined for every time step
    defd_mask = numpy.logical_and.reduce(defined_data, axis=pyferret.T_AXIS)
    for t in xrange(ntime):    
        defined_data[:, :, :, t] = defd_mask 
    # Convert to time-location (a 2-D array), 
    # eliminating locations with missing time steps, 
    # and taking the transpose so time is the first axis
    timeloc = inputs[pyferret.ARG1][defined_data]\
                    .reshape((-1, ntime)).T
    # Create the EOFAnalysis object and analyze the data
    eofs = eofanal.EOFAnalysis(timeloc)
    eofs.setminsignif(min_signif)
    eofs.analyze()
    # Initialize the result to all-undefined
    result[:] = result_bdf
    # Remove the E and F singleton axes from the defined mask
    defined_data = defined_data.reshape(defined_data.shape[:4])
    # Assign the EOF-TAF products for the significant EOFs 
    # The values at m=0 are the time-series averages
    numeofs = eofs.numeofs()
    for k in xrange(numeofs+1):
        timeloc_piece = eofs.datapiece(k)
        loctime_piece = numpy.array(timeloc_piece.T).reshape(-1)
        result[:, :, :, :, k, 0][defined_data] = loctime_piece
    # The EOF-TAF pieces for insignificant EOFs are left as undefined 
    return


if __name__ == "__main__":
    # Just verify ferret_init does not raise an error
    defdict = ferret_init(0)
    # Create some data
    yaxis = numpy.linspace(-80.0, 80.0, 17)
    zaxis = numpy.linspace(0.0, 250.0, 6)
    taxis = numpy.linspace(0.0, 8760.0, 25)
    ydata = 20.0 * numpy.square(numpy.cos(numpy.deg2rad(yaxis)))
    zdata = 1.0 / numpy.log10(5.0 * zaxis + 10.0)
    yzdata = numpy.outer(ydata, zdata)
    tdata = (taxis / 87600.0) + 1.0
    yztdata = numpy.outer(yzdata, tdata).reshape((1, 17, 6, 25, 1, 1))
    ydata = numpy.square(numpy.sin(2.0 * numpy.deg2rad(yaxis)))
    yzdata = numpy.outer(ydata, zdata)
    tdata = -5.0 * numpy.cos((tdata - 2190.0) * numpy.pi / 4380.0)
    yztdata += numpy.outer(yzdata, tdata).reshape((1, 17, 6, 25, 1, 1))
    yztdata += 6.0
    print "time series at Y = 0.0, Z = 0.0"
    print str(yztdata[0, 8, 0, :, 0, 0])
    print "depth series at Y = 0.0, T = start of April"
    print str(yztdata[0, 8, :, 6, 0, 0])
    print "latitude series at Z = 0.0, T = start of April"
    print str(yztdata[0, :, 0, 6, 0, 0])
    # Create the result array and the other ferret_compute arguments
    result = numpy.zeros((1, 17, 6, 25, 17*6, 1))
    resbdf = numpy.array([1.0E20])
    inputs = (yztdata, 0.001)
    inpbdfs = (-1.0E34, -1.0E34)
    # Check ferret_compute
    ferret_compute(0, result, resbdf, inputs, inpbdfs)
    piecesum = numpy.zeros((1, 17, 6, 25))
    lastone = 0
    for m in xrange(17 * 6):
        if numpy.allclose(result[:, :, :, :, m, 0], resbdf):
            break;
        print "EOF-TAF piece %d" % m
        print "    time series at Y = 0.0, Z = 0.0"
        print "    " + str(result[0, 8, 0, :, m, 0])
        print "    depth series at Y = 0.0, T = start of April"
        print "    " + str(result[0, 8, :, 6, m, 0])
        print "    latitude series at Z = 0.0, T = start of April"
        print "    " + str(result[0, :, 0, 6, m, 0])
        lastone = m
        piecesum += result[:, :, :, :, m, 0]
    if numpy.allclose(piecesum, yztdata[:, :, :, :, 0, 0]):
        print "sum of %d pieces all close to input data" % lastone
    else:
        print "sum %d pieces different from input data" % lastone

