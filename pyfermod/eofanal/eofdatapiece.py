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
    init_dict["descript"] = \
        "Partitions data into EOF * TAF pieces along the ensemble axis"
    init_dict["argnames"] = ("Data",
                             "MinSignif")
    init_dict["argdescripts"] = (
        "Time-location data; defined on regular T and one or more of X, Y, Z",
        "Minimum fraction-of-data-explained to be considered significant")
    init_dict["argtypes"] = (pyferret.FLOAT_ARRAY,
                             pyferret.FLOAT_ONEVAL)
    # X, Y, Z, and T match input; E axis added as an abstract axis
    axes = [ pyferret.AXIS_IMPLIED_BY_ARGS ] * pyferret.MAX_FERRET_NDIM
    axes[pyferret.E_AXIS] = pyferret.AXIS_ABSTRACT
    axes[pyferret.F_AXIS] = pyferret.AXIS_DOES_NOT_EXIST;
    init_dict["axes"] = axes
    no_influence = [ False ] * pyferret.MAX_FERRET_NDIM
    part_influence = [ True ] * pyferret.MAX_FERRET_NDIM
    part_influence[pyferret.E_AXIS] = False
    part_influence[pyferret.F_AXIS] = False
    init_dict["influences"] = (part_influence,
                               no_influence)
    init_dict["piecemeal"] =  [ False ] * pyferret.MAX_FERRET_NDIM

    return init_dict


def ferret_result_limits(efid):
    '''
    Provides the bounds of the E abstract axis
    '''
    time_axis_info = pyferret.get_axis_info(efid, pyferret.ARG1, pyferret.T_AXIS);
    ntime = time_axis_info.get("size", -1)
    if ntime < 0:
        raise ValueError("The time axis of the input data is not bounded (not pre-defined)")
    if ntime < 2:
        raise ValueError("Unexpectedly small number of time steps (%d) in the input data" % ntime)
    regular = time_axis_info.get("regular", False)
    if not regular:
        raise ValueError("The time axis of the input data is not a regularly-spaced axis")
    result_limits = [ None ] * pyferret.MAX_FERRET_NDIM
    result_limits[pyferret.E_AXIS] = (0, ntime)
    return result_limits


def ferret_compute(efid, result, result_bdf, inputs, input_bdfs):
    '''
    '''
    # verify no ensemble or forecast axis on the input data
    if inputs[pyferret.ARG1].shape[pyferret.E_AXIS] > 1:
        raise ValueError("Input data cannot have an ensemble axis")
    if inputs[pyferret.ARG1].shape[pyferret.F_AXIS] > 1:
        raise ValueError("Input data cannot have a forecast axis")
    # Verify the second value is reasonable
    min_signif = float(inputs[1])
    if (min_signif < 1.0E-6) or (min_signif > (1.0 - 1.0E-6)):
        raise ValueError("MinSignif must be in [0.000001, 0.999999]")
    # number of time steps in the input data
    ntime = inputs[pyferret.ARG1].shape[pyferret.T_AXIS]
    # Get the mask of where the data is defined
    defined_data = ( numpy.fabs(inputs[pyferret.ARG1] - 
                                input_bdfs[pyferret.ARG1]) >= 1.0E-5 )
    # Get the mask of where the data is defined for every time step
    defd_mask = numpy.logical_and.reduce(defined_data, axis=pyferret.T_AXIS)
    for t in xrange(ntime):    
        defined_data[:,:,:,t] = defd_mask 
    # Convert to time-location (a 2-D array), eliminating locations with missing time steps
    # The transpose is used so the time axis is the first axis. 
    timeloc = inputs[pyferret.ARG1][defined_data].reshape((-1, ntime)).T
    # Create the EOFAnalysis object and analyze the data
    eofanal = eofanal.EOFAnalysis(timeloc)
    eofanal.setminsignif(min_signif)
    eofanal.analyze()
    # Initialize the result to all-undefined
    result[:] = result_bdf
    # Assign the EOF-TAF products for the significant EOFs to the result
    # The values at m=0 are the time-series averages
    numeofs = eofanal.numeofs()
    for k in xrange(0, numeofs):
        timeloc_piece = eofanal.datapiece(k)
        result[:,:,:,:,k][defined_data] = timeloc_piece.T
    # The EOF-TAF products for insignificant EOFs are left as undefined 
    return
