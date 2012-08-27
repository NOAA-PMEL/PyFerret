'''
PyFerret Python External Function (PyEF) for regridding data from
a curvilinear longitude, latitude grid to a rectilinear longitude,
latitude grid.  Uses the Ferret2DRegridder class, which in turn
uses ESMP/ESMF, to perform the regridding.

@author: Karl Smith
'''
import numpy
import pyferret
import ESMP
import pyferret.regrid as regrid


def ferret_init(efid):
    '''
    Initializes the curv2rect function.  Either center or corner curvilinear
    coordinates must be given; both may be given.  If only corner coordinates
    are given, center coordinates are positioned at the centroid of each cell
    as computed using the Surveyor's Formula.  If conservative regridding is
    used, corners must be given.
    '''
    init_dict = { }
    init_dict["numargs"] = 7
    init_dict["descript"] = "Regrids data from curvilinear lon,lat center " \
                            "and/or corner grid to rectilinear using ESMP/ESMF"
    init_dict["argnames"] = ("CurvData",
                              "CurvCenterLons", "CurvCenterLats",
                              "CurvCornerLons", "CurvCornerLats",
                              "TemplateRectVar",
                              "Method")
    init_dict["argdescripts"] = ("Curvilinear X,Y data positioned at centers",
                                  "Curvilinear center longitudes on an X,Y grid",
                                  "Curvilinear center latitudes on an X,Y grid",
                                  "Curvilinear corner longitudes on an X,Y (maybe Z) grid",
                                  "Curvilinear corner latitudes on an X,Y (maybe Z) grid",
                                  "Template variable on the desired rectilinear grid",
                                  "Regrid method: BILINEAR, PATCH, CONSERVE")
    init_dict["argtypes"] = (pyferret.FLOAT_ARRAY, pyferret.FLOAT_ARRAY,
                              pyferret.FLOAT_ARRAY, pyferret.FLOAT_ARRAY,
                              pyferret.FLOAT_ARRAY, pyferret.FLOAT_ARRAY,
                              pyferret.STRING_ONEVAL)
    init_dict["influences"] = ((False, False, True, True, True, True),
                                (False, False, False, False, False, False),
                                (False, False, False, False, False, False),
                                (False, False, False, False, False, False),
                                (False, False, False, False, False, False),
                                (True, True, True, True, True, True),
                                (False, False, False, False, False, False))
    return init_dict


def ferret_compute(efid, result, result_bdf, inputs, input_bdfs):
    '''
    Performs the regridding for the curv2rect function.

    Arguments:
        result     - rectilinear data values to be assigned
        result_bdf - missing-data value for result
        inputs     - (CurvCenterLons, CurvCenterLats,
                      CurvConerLons, CurvCornerLats.
                      CurvData, TemplateRectVar, Method)
        input_bdfs - missing-data values for the corresponding inputs array
    '''
    # Get the regridding method to use
    methodstr = pyferret.get_arg_one_val(efid, pyferret.ARG7).upper()
    if methodstr == "BILINEAR":
        method = ESMP.ESMP_REGRIDMETHOD_BILINEAR
    elif methodstr == "PATCH":
        method = ESMP.ESMP_REGRIDMETHOD_PATCH
    elif methodstr == "CONSERVE":
        method = ESMP.ESMP_REGRIDMETHOD_CONSERVE
    else:
        raise ValueError("Unknown method %s" % methodstr)

    # Get the template data and missing value
    template_data = inputs[pyferret.ARG6]
    template_undef = input_bdfs[pyferret.ARG6]

    # Get the rectilinear longitude edges
    lo, hi = pyferret.get_axis_box_limits(efid, pyferret.ARG6, pyferret.X_AXIS)
    if not numpy.allclose(lo[1:], hi[:-1]):
        raise ValueError("Unexpected ARG6 X_AXIS box limit values returned from pyferret.get_axis_box_limits")
    rect_edge_lons = numpy.empty((lo.shape[0] + 1), dtype=numpy.float64)
    rect_edge_lons[0] = lo[0]
    rect_edge_lons[1:] = hi

    # Get the rectilinear latitude edges
    lo, hi = pyferret.get_axis_box_limits(efid, pyferret.ARG6, pyferret.Y_AXIS)
    if not numpy.allclose(lo[1:], hi[:-1]):
        raise ValueError("Unexpected ARG6 Y_AXIS box limitvalues returned from pyferret.get_axis_box_limits")
    rect_edge_lats = numpy.empty((lo.shape[0] + 1), dtype=numpy.float64)
    rect_edge_lats[0] = lo[0]
    rect_edge_lats[1:] = hi

    # Get the curvilinear data
    curv_data = inputs[pyferret.ARG1]
    curv_centers_shape = (curv_data.shape[0], curv_data.shape[1])
    if (curv_data.shape[2:] != template_data.shape[2:]):
        raise ValueError("Curvilinear data and template variable must share Z, T, E, and F axes")
    curv_undef = input_bdfs[pyferret.ARG1]

    # Get the curvilinear centers arrays
    lons = inputs[pyferret.ARG2].squeeze()
    lats = inputs[pyferret.ARG3].squeeze()
    if (lons.shape == curv_centers_shape) and (lats.shape == curv_centers_shape):
        curv_center_lons = lons
        curv_center_lats = lats
    elif (lons.shape == ()) and (lats.shape == ()):
        curv_center_lons = None
        curv_center_lats = None
    else:
        raise ValueError("Curvilinear center points must have appropriate shape or be singletons")

    # Get the curvilinear corners arrays
    lons = inputs[pyferret.ARG4].squeeze()
    lats = inputs[pyferret.ARG5].squeeze()
    corners_shape = (curv_data.shape[0] + 1, curv_data.shape[1] + 1)
    corners_shape_3d = (curv_data.shape[0], curv_data.shape[1], 4)
    if (lons.shape == corners_shape) and (lats.shape == corners_shape):
        curv_corner_lons = lons
        curv_corner_lats = lats
    elif (lons.shape == corners_shape_3d) and (lats.shape == corners_shape_3d):
        curv_corner_lons, curv_corner_lats = regrid.quadCornersFrom3D(lons, lats)
    elif method == ESMP.ESMP_REGRIDMETHOD_CONSERVE:
        raise ValueError("CONSERVE method requires curvilinear grid corner coordinates")
    elif (lons.shape == ()) and (lats.shape == ()):
        curv_corner_lons = None
        curv_corner_lats = None
    else:
        raise ValueError("Curvilinear corner points must have appropriate shape or be singletons")

    # If curvilinear center point coordinates not given, 
    # generate them from the curvilienar corner points
    if curv_center_lons is None:
        if not curv_corner_lons is None:
            curv_center_lons, curv_center_lats = \
                regrid.quadCentroids(curv_corner_lons, curv_corner_lats)
        else:
            raise ValueError("Valid center or corner curvilinear grid coordinates must be given")

    # Make sure ESMP is, or has been, initialized
    regrid.ESMPControl().startCheckESMP()

    # Create the regridder used repeatedly in this function 
    regridder = regrid.CurvRectRegridder()
    last_curv_center_ignore = None
    last_rect_center_ignore = None

    # Increment the depth index last - most likely to change the undefined (e.g., land) mask 
    for d_idx in xrange(curv_data.shape[2]):
        # Arrays are probably in Fortran order, so increment last indices last
        for f_idx in xrange(curv_data.shape[5]):
            for e_idx in xrange(curv_data.shape[4]):
                for t_idx in xrange(curv_data.shape[3]):
                    curv_center_ignore = (curv_data[:, :, d_idx, t_idx, e_idx, f_idx] == curv_undef)
                    if (last_curv_center_ignore is None) or \
                       numpy.any(curv_center_ignore != last_curv_center_ignore):
                        regridder.createCurvGrid(curv_center_lons, curv_center_lats, curv_center_ignore,
                                                 curv_corner_lons, curv_corner_lats, None)
                        last_curv_center_ignore = curv_center_ignore
                    rect_center_ignore = (template_data[:, :, d_idx, t_idx, e_idx, f_idx] == template_undef)
                    if (last_rect_center_ignore is None) or \
                       numpy.any(rect_center_ignore != last_rect_center_ignore):
                        regridder.createRectGrid(rect_edge_lons, rect_edge_lats, rect_center_ignore, None)
                        last_rect_center_ignore = rect_center_ignore
                    regridder.assignCurvField(curv_data[:, :, d_idx, t_idx, e_idx, f_idx])
                    regridder.assignRectField(None)
                    regrid_data = regridder.regridCurvToRect(result_bdf, method)
                    result[:, :, d_idx, t_idx, e_idx, f_idx] = regrid_data

    return

