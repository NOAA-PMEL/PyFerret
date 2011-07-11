"""
Creates a shapefile with a given root name using data from a given value 
array.  The shapes are quadrilaterals in the X,Y-plane at a given Z (if 
this axis is used).  The vertices of the quadrilaterals are the values 
of the bounding boxes of the X and Y axes of the given value array.  The 
value(s) associated with each shape comes from the value array.  If the 
T axis is used, each T-axis coordinate generates a field in the database
associated with the shapefile.
"""

import pyferret
import shapefile

def ferret_init(efid):
    """
    Initialization for the shapefile_writeval PyEF
    """
    retdict = { "numargs": 2,
                "descript": "Writes a shapefile of XY quadrilaterals that are associated with the given values",
                "restype": pyferret.FLOAT_ARRAY,
                "axes": ( pyferret.AXIS_ABSTRACT,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST, ),
                "argnames": ( "SHAPEFILE", "VALUE", ),
                "argdescripts": ( "Name for the shapefile (any extension given is ignored)",
                                  "Value(s) for the shapes; must have X and Y axes, Z and T axes optional", ),
                "argtypes": ( pyferret.STRING_ONEVAL,
                              pyferret.FLOAT_ARRAY, ),
                "influences": ( (False, False, False, False),
                                (False, False, False, False), ),
              }
    return retdict


def ferret_result_limits(efid):
    """
    Abstract axis limits for the shapefile_writeval PyEF
    """
    return ( (1, 1), None, None, None, )


def ferret_compute(efid, result, resbdf, inputs, inpbdfs):
    """
    Create the shapefile named in inputs[0] using the values array given
    in inputs[1].  The bounding box limits of the X and Y axes of inputs[1] 
    are used to create the quadrilaterals.  If given, Z axis values will 
    be included in the shapes coordinates and generate layers of shapes.
    If given, T axis values will generate field names in the database of
    values associated with each shape.  The values in inputs[1] is then
    used as the values associated with each shape.
    If successful, fills result (which might as well be a 1x1x1x1 array)
    with zeros.  If problems, an error will be raised.
    """
    shapefile_name = inputs[0]
    shapefile_values = inputs[1]
    # Get the X and Y axis box limits for the quadrilateral coordinates
    x_box_limits = pyferret.get_axis_box_limits(efid, pyferret.ARG2, pyferret.X_AXIS)
    if x_box_limits == None:
        raise ValueError("Unable to determine the X axis box limits")
    num_xs = len(x_box_limits) - 1
    y_box_limits = pyferret.get_axis_box_limits(efid, pyferret.ARG2, pyferret.Y_AXIS)
    if y_box_limits == None:
        raise ValueError("Unable to determine the y axis box limits")
    num_ys = len(y_box_limits) - 1
    # Get the elevation/depth coordinates
    z_coords = pyferret.get_axis_coordinates(efid, pyferret.ARG2, pyferret.Z_AXIS)
    # z_coords can be None
    # Assign the appropriate type shapes
    if z_coords == None:
        shapetype = shapefile.POLYGON
        num_zs = 0
    else:
        shapetype = shapefile.POLYGONZ
        num_zs = len(z_coords)
    # Create the shapefile object
    sf = shapefile.Writer(shapetype)
    # Time coordinates become 
    # TODO: need get_time_axis_coordinates in order to generate meaningful values
    t_coords = pyferret.get_axis_coordinates(efid, pyferret.ARG2, pyferret.T_AXIS)
    # t_coords can be None
    # Add the fields to the associated database
    # TODO: get reasonable names and sizes for the fields
    if t_coords == None:
        sf.field("Value", "N", 20, 7)
    else:
        for t in t_coords:
            sf.field("T%d" % t, "N", 20, 7)
    # Write out the shapes and the records
    if z_coords == None:
        for j in xrange(num_ys):
            ylow = y_box_limits[j]
            yhigh = y_box_limits[j+1]
            if ylow > yhigh:
                ylow, yhigh = yhigh, ylow
            for i in xrange(num_xs):
                xlow = x_box_limits[i]
                xhigh = x_box_limits[i+1]
                if xlow > xhigh:
                    xlow, xhigh = xhigh, xlow
                sf.poly([ [ [xlow, ylow],   [xlow, yhigh], [xhigh, yhigh], 
                            [xhigh, ylow],  [xlow, ylow] ] ], shapetype)
                sf.record(shapefile_values[i, j, 0, :])
    else:
        for k in xrange(num_zs):
            zval = z_coords[k]
            for j in xrange(num_ys):
                ylow = y_box_limits[j]
                yhigh = y_box_limits[j+1]
                if ylow > yhigh:
                    ylow, yhigh = yhigh, ylow
                for i in xrange(num_xs):
                    xlow = x_box_limits[i]
                    xhigh = x_box_limits[i+1]
                    if xlow > xhigh:
                        xlow, xhigh = xhigh, xlow
                    sf.poly([ [ [xlow, ylow, zval],   [xlow, yhigh, zval], [xhigh, yhigh, zval], 
                                [xhigh, ylow, zval],  [xlow, ylow, zval] ] ], shapetype)
                    sf.record(shapefile_values[i, j, k, :])
    sf.save(shapefile_name)
    result[:, :, :, :] = 0


#
# Difficult to create a unit test for this module since it uses pyferret.get_axis_....
#
