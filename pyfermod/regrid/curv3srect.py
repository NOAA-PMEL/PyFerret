'''
PyFerret Python External Function (PyEF) for regridding data from
a curvilinear longitude, latitude, bathymetry/zeta/sigma grid to
a rectilinear, longitude, latitude, depth grid.  Uses the
Ferret3DRegridder class, which in turn uses ESMP/ESMF, to perform
the regridding.

@author: Karl Smith
'''

import numpy
import pyferret
import ESMP
import pyferret.regrid as regrid


def ferret_init(efid):
    '''
    Initializes the curv3srect function.  The curvilinear data is assumed
    to be positioned at the curvilinear coordinates provided.
    '''
    init_dict = { }
    init_dict["numargs"] = 7
    init_dict["descript"] = \
        "Regrids data from curvilinear lon, lat, sigma, bathymetry, zeta " \
        "(centers) grid to rectilinear lon, lat, depth using ESMP/ESMF"
    init_dict["argnames"] = ("CurvData",
                             "CurvLons",
                             "CurvLats",
                             "CurvBaths",
                             "CurvZetas",
                             "TemplateRectVar",
                             "Method")
    init_dict["argdescripts"] = (
        "Curvilinear X,Y,Z,[T,E,F] data where Z is sigma values",
        "Longitudes of curvilinear data on an X,Y grid",
        "Latitudes of curvilinear data on an X,Y grid",
        "Bathymetry (as depths) of curvilinear data on an X,Y grid",
        "Water surface elevations of curvilinear data on an X,Y,[T] grid (optional)",
        "Template variable on the desired rectilinear X,Y,Z,[T,E,F] grid " \
            "where Z is depths",
        "Regrid method: BILINEAR, PATCH")
    init_dict["argtypes"] = (pyferret.FLOAT_ARRAY,
                             pyferret.FLOAT_ARRAY,
                             pyferret.FLOAT_ARRAY,
                             pyferret.FLOAT_ARRAY,
                             pyferret.FLOAT_ARRAY,
                             pyferret.FLOAT_ARRAY,
                             pyferret.STRING_ONEVAL)
    no_influence = [ False ] * pyferret.MAX_FERRET_NDIM
    full_influence = [ True ] * pyferret.MAX_FERRET_NDIM
    part_influence = full_influence[:]
    part_influence[pyferret.X_AXIS] = False
    part_influence[pyferret.Y_AXIS] = False
    part_influence[pyferret.Z_AXIS] = False
    time_influence = no_influence[:]
    time_influence[pyferret.T_AXIS] = True
    init_dict["influences"] = (part_influence,
                               no_influence,
                               no_influence,
                               no_influence,
                               time_influence,
                               full_influence,
                               no_influence)
    # okay to break up along T, E, and F result axes
    init_dict["piecemeal"] = (False, False, False, True, True, True)

    return init_dict


def ferret_compute(efid, result, result_bdf, inputs, input_bdfs):
    '''
    Performs the regridding for the curv3srect function.

    Arguments:
        result     - rectilinear data values to be assigned
        result_bdf - missing-data value for result
        inputs     - (CurvData, CurvLons, CurvLats, CurvBaths,
                      CurvZetas, TemplateRectVar, Method)
        input_bdfs - missing-data values for the corresponding inputs array
    '''
    # Get the regridding method to use
    methodstr = pyferret.get_arg_one_val(efid, pyferret.ARG7).upper()
    if methodstr == "BILINEAR":
        method = ESMP.ESMP_REGRIDMETHOD_BILINEAR
    elif methodstr == "PATCH":
        method = ESMP.ESMP_REGRIDMETHOD_PATCH
    else:
        raise ValueError("Unknown method %s (CONSERVE not supported)" % methodstr)

    # Get the template data and missing value
    template_data = inputs[pyferret.ARG6]
    template_undef = input_bdfs[pyferret.ARG6]

    # Get the rectilinear center longitudes, latitudes, and depths
    rect_center_lons = pyferret.get_axis_coordinates(efid,
                                    pyferret.ARG6, pyferret.X_AXIS)
    rect_center_lats = pyferret.get_axis_coordinates(efid,
                                    pyferret.ARG6, pyferret.Y_AXIS)
    rect_center_depths = pyferret.get_axis_coordinates(efid,
                                      pyferret.ARG6, pyferret.Z_AXIS)

    # Get the rectilinear corner longitudes
    lo, hi = pyferret.get_axis_box_limits(efid, pyferret.ARG6, pyferret.X_AXIS)
    if not numpy.allclose(lo[1:], hi[:-1]):
        raise ValueError("Unexpected ARG6 X_AXIS box limit values " \
                         "returned from pyferret.get_axis_box_limits")
    rect_corner_lons = numpy.empty((lo.shape[0] + 1), dtype=numpy.float64)
    rect_corner_lons[0] = lo[0]
    rect_corner_lons[1:] = hi

    # Get the rectilinear corner latitudes
    lo, hi = pyferret.get_axis_box_limits(efid, pyferret.ARG6, pyferret.Y_AXIS)
    if not numpy.allclose(lo[1:], hi[:-1]):
        raise ValueError("Unexpected ARG6 Y_AXIS box limit values " \
                         "returned from pyferret.get_axis_box_limits")
    rect_corner_lats = numpy.empty((lo.shape[0] + 1), dtype=numpy.float64)
    rect_corner_lats[0] = lo[0]
    rect_corner_lats[1:] = hi

    # Get the rectilinear corner depths
    lo, hi = pyferret.get_axis_box_limits(efid, pyferret.ARG6, pyferret.Z_AXIS)
    if not numpy.allclose(lo[1:], hi[:-1]):
        raise ValueError("Unexpected ARG6 Z_AXIS box limit values " \
                         "returned from pyferret.get_axis_box_limits")
    rect_corner_depths = numpy.empty((lo.shape[0] + 1), dtype=numpy.float64)
    rect_corner_depths[0] = lo[0]
    rect_corner_depths[1:] = hi

    # Get the curvilinear data
    curv_data = inputs[pyferret.ARG1]
    if curv_data.shape[3:] != template_data.shape[3:]:
        raise ValueError("Curvilinear data and template variable " \
                         "must have same T, E, and F axes")
    curv_undef = input_bdfs[pyferret.ARG1]

    # Get the curvilinear centers arrays - squeeze removes the singleton axes
    curv_center_lons  = inputs[pyferret.ARG2].squeeze()
    curv_center_lats  = inputs[pyferret.ARG3].squeeze()
    curv_center_baths = inputs[pyferret.ARG4].squeeze()
    curv_centers_shape = curv_data.shape[:2]
    if (curv_center_lons.shape  != curv_centers_shape) or \
       (curv_center_lats.shape  != curv_centers_shape) or \
       (curv_center_baths.shape != curv_centers_shape):
        raise ValueError("Curvilinear data, longitude, latitudes, and " \
                         "and bathymetry must have same X and Y axes")

    # Squeeze should remove a singleton Z axis in zetas
    curv_center_zetas = inputs[pyferret.ARG5].squeeze()
    # If only one time step, squeeze would have also removed it.
    # So if no time axis, put one in.
    if len(curv_center_zetas.shape) == 2:
        curv_center_zetas = curv_center_zetas[:,:,numpy.newaxis]
    # Allow zeta to be omitted by giving a single-point array
    if curv_center_zetas.shape == ():
        curv_center_zetas = None
    elif curv_center_zetas.shape != (curv_data.shape[0],
                                     curv_data.shape[1],
                                     curv_data.shape[3]):
        raise ValueError("Curvilinear data and zetas " \
                         "must have same X, Y, and T axes")

    # Get the sigma values from the Z axis of curv_data
    curv_center_sigmas = pyferret.get_axis_coordinates(efid,
                                      pyferret.ARG1, pyferret.Z_AXIS)

    curv_centers_shape = curv_data.shape[:3]
    # Expand the sigmas to 3D (adding X and Y axes) to simplify
    # calculation of curvilinear depths
    curv_center_sigmas = numpy.repeat(curv_center_sigmas,
                                      curv_centers_shape[0] * \
                                      curv_centers_shape[1]) \
                              .reshape(curv_centers_shape, order='F')
    # Expand the curvilinear longitude, latitude, and bathymetry
    # arrays to 3D (adding Z axis)
    curv_center_lons = numpy.tile(curv_center_lons.flatten('F'),
                                  curv_centers_shape[2]) \
                            .reshape(curv_centers_shape, order='F')
    curv_center_lats = numpy.tile(curv_center_lats.flatten('F'),
                                  curv_centers_shape[2]) \
                            .reshape(curv_centers_shape, order='F')
    curv_center_baths = numpy.tile(curv_center_baths.flatten('F'),
                                   curv_centers_shape[2]) \
                             .reshape(curv_centers_shape, order='F')

    # Make sure ESMP is, or has been, initialized
    regrid.ESMPControl().startCheckESMP()

    # Create the regridder used repeatedly in this function 
    regridder3d = regrid.CurvRect3DRegridder()
    last_rect_center_ignore = None

    if curv_center_zetas == None:
        # Create the curvilinear depths array
        curv_center_depths = curv_center_sigmas * curv_center_baths
        last_curv_center_ignore = None
        
    # Increment the time index last since zeta is time dependent
    for t_idx in range(curv_data.shape[3]):

        if curv_center_zetas != None:
            # Expand the zetas for this time step to 3D - adding Z axis
            zetas = numpy.tile(curv_center_zetas[:,:,t_idx].flatten('F'),
                               curv_centers_shape[2]) \
                         .reshape(curv_centers_shape, order='F')
            # Create the curvilinear depths array
            curv_center_depths = curv_center_sigmas * (curv_center_baths + \
                                                       zetas) - zetas
            # Different curvilinear depths, so need to recreate the curvilinear grid
            last_curv_center_ignore = None

        # Arrays are probably in Fortran order, so increment last indices last
        for f_idx in range(curv_data.shape[5]):
            for e_idx in range(curv_data.shape[4]):
                # Determine curvilinear center points to ignore from undefined data
                curv_center_ignore = ( numpy.abs(curv_data[:, :, :, t_idx,
                                        e_idx, f_idx] - curv_undef) < 1.0E-7 )
                # If mask has changed, need to recreate the curvilinear grid
                if (last_curv_center_ignore is None) or \
                   numpy.any(curv_center_ignore != last_curv_center_ignore):
                    regridder3d.createCurvGrid(curv_center_lons, curv_center_lats,
                                               curv_center_depths, curv_center_ignore,
                                               True, None, None, None, None)
                    last_curv_center_ignore = curv_center_ignore
                # Determine rectilinear center points to ignore from undefined data
                rect_center_ignore = ( numpy.abs(template_data[:, :, :, t_idx,
                                        e_idx, f_idx] - template_undef) < 1.0E-7 )
                # If mask has changed, need to recreate the rectilinear grid
                if (last_rect_center_ignore is None) or \
                   numpy.any(rect_center_ignore != last_rect_center_ignore):
                    regridder3d.createRectGrid(rect_center_lons, rect_center_lats,
                                               rect_center_depths, rect_center_ignore,
                                               True,
                                               rect_corner_lons, rect_corner_lats,
                                               rect_corner_depths, None)
                    last_rect_center_ignore = rect_center_ignore
                # Assign the curvilinear data
                regridder3d.assignCurvField(curv_data[:, :, :, t_idx, e_idx, f_idx])
                # Allocate space for the rectilinear data from the regridding
                regridder3d.assignRectField(None)
                # Regrid and assign the rectilinear data to the results array
                regrid_data = regridder3d.regridCurvToRect(result_bdf, method)
                result[:, :, :, t_idx, e_idx, f_idx] = regrid_data

    return

