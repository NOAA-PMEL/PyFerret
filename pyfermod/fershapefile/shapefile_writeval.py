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
import pyferret.fershapefile
import shapefile

def ferret_init(efid):
    """
    Initialization for the shapefile_writeval PyEF
    """
    retdict = { "numargs": 3,
                "descript": "Writes a shapefile of XY quadrilaterals that are associated with the given values",
                "restype": pyferret.FLOAT_ARRAY,
                "axes": ( pyferret.AXIS_ABSTRACT,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST, ),
                "argnames": ( "SHAPEFILE", "VALUE", "MAPPRJ"),
                "argdescripts": ( "Name for the shapefile (any extension given is ignored)",
                                  "Value(s) for the shapes; must have X and Y axes, Z and T axes optional",
                                  "Either a common name or a WKT description for the map projection; " \
                                      "if blank, WGS 84 is used", ),
                "argtypes": ( pyferret.STRING_ONEVAL,
                              pyferret.FLOAT_ARRAY,
                              pyferret.STRING_ONEVAL, ),
                "influences": ( (False, False, False, False),
                                (False, False, False, False),
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
    used as the values associated with each shape.  Either a common name
    or a WKT description of the map projection for the coordinated should
    be given in inputs[2].  If blank, WGS 84 is used.  If successful,
    fills result (which might as well be a 1x1x1x1 array) with zeros.
    If problems, an error will be raised.
    """
    shapefile_name = inputs[0]
    shapefile_values = inputs[1]
    shapefile_mapprj = inputs[2]
    # Get the X and Y axis box limits for the quadrilateral coordinates
    x_box_limits = pyferret.get_axis_box_limits(efid, pyferret.ARG2, pyferret.X_AXIS)
    if x_box_limits == None:
        raise ValueError("Unable to determine the X axis box limits")
    lowerxs = x_box_limits[0]
    upperxs = x_box_limits[1]
    y_box_limits = pyferret.get_axis_box_limits(efid, pyferret.ARG2, pyferret.Y_AXIS)
    if y_box_limits == None:
        raise ValueError("Unable to determine the y axis box limits")
    lowerys = y_box_limits[0]
    upperys = y_box_limits[1]
    # Get the elevation/depth coordinates, or None if they do not exist
    z_coords = pyferret.get_axis_coordinates(efid, pyferret.ARG2, pyferret.Z_AXIS)
    # Create the shapefile object with the appropriate shape type
    if z_coords == None:
        sfwriter = shapefile.Writer(shapefile.POLYGON)
    else:
        sfwriter = shapefile.Writer(shapefile.POLYGONZ)
    # Time coordinates, if they exist, are used for field names of values associated with each shape
    # TODO: need get_time_axis_coordinates in order to generate meaningful values
    t_coords = pyferret.get_axis_coordinates(efid, pyferret.ARG2, pyferret.T_AXIS)
    # Create the field(s) for the values
    # TODO: get reasonable names and sizes for the fields
    if t_coords == None:
        sfwriter.field("Value", "N", 20, 7)
    else:
        for t in t_coords:
            sfwriter.field("T%d" % t, "N", 20, 7)
    # Write out the shapes and the records
    if z_coords == None:
        for j in xrange(len(lowerys)):
            for i in xrange(len(lowerxs)):
                pyferret.fershapefile.addquadxyvalues(sfwriter,
                         ( lowerxs[i], lowerys[j] ),
                         ( lowerxs[i], upperys[j] ),
                         ( upperxs[i], upperys[j] ),
                         ( upperxs[i], lowerys[j] ),
                         None, shapefile_values[i, j, 0, :].tolist())
    else:
        for k in xrange(len(z_coords)):
            for j in xrange(len(lowerys)):
                for i in xrange(len(lowerxs)):
                    pyferret.fershapefile.addquadxyvalues(sfwriter,
                             ( lowerxs[i], lowerys[j] ),
                             ( lowerxs[i], upperys[j] ),
                             ( upperxs[i], upperys[j] ),
                             ( upperxs[i], lowerys[j] ),
                             z_coords[k], shapefile_values[i, j, k, :].tolist())
    sfwriter.save(shapefile_name)
    # Create the .prj file from the map projection common name or the WKT description
    pyferret.fershapefile.createprjfile(shapefile_mapprj, shapefile_name)
    result[:, :, :, :] = 0


#
# The following is only for testing this module from the command line
#
if __name__ == "__main__":
    # make sure these calls do not generate errors
    info = ferret_init(0)
    del info
    limits = ferret_result_limits(0)
    del limits
    # Testing ferret_compute difficult due to call
    # to get_axis_box_limits and get_axis_coordinates

    print "shapefile_writeval: SUCCESS (limited)"
