"""
Creates a shapefile with a given root name using data from a given value
array.  The shapes are quadrilaterals in the X,Y-plane at a given Z (if
this axis is used).  The vertices of the quadrilaterals are the values
of the bounding boxes of the X and Y axes of the given value array.  The
value(s) associated with each shape comes from the value array.
"""

import pyferret
import pyferret.fershp
import shapefile

def ferret_init(efid):
    """
    Initialization for the shapefile_writeval PyEF
    """
    retdict = { "numargs": 4,
                "descript": "Writes a shapefile of XY quadrilaterals associated with given values",
                "restype": pyferret.FLOAT_ARRAY,
                "axes": ( pyferret.AXIS_ABSTRACT,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST, ),
                "argnames": ( "SHAPEFILE", "VALUE", "VALNAME", "MAPPRJ"),
                "argdescripts": ( "Shapefile name (any extension given is ignored)",
                                  "Shape values; X and Y axes required, " \
                                      "Z axis optional, T axis undefined or singleton",
                                  "Name for the shape value",
                                  "Common name or WKT description of map projection; " \
                                      "if blank, WGS 84 is used", ),
                "argtypes": ( pyferret.STRING_ONEVAL,
                              pyferret.FLOAT_ARRAY,
                              pyferret.STRING_ONEVAL,
                              pyferret.STRING_ONEVAL, ),
                "influences": ( (False, False, False, False),
                                (False, False, False, False),
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
    Create the shapefile named in inputs[0] using the values array
    given in inputs[1].  The bounding box limits of the X and Y axes
    of inputs[1] are used to create the quadrilaterals.  The values
    in inputs[1] are used as the values associated with each shape
    using the field name given in inputs[2].  Either a common name
    or a WKT description of the map projection for the coordinated
    should be given in inputs[3].  If blank, WGS 84 is used.  If
    successful, fills result (which might as well be a 1x1x1x1 array)
    with zeros.  If a problem occurs, an error will be raised.
    """
    shapefile_name = inputs[0]
    values = inputs[1]
    missing_value = inpbdfs[1]
    field_name = inputs[2].strip()
    if not field_name:
        field_name = "VALUE"
    map_projection = inputs[3]

    # Verify the shapes are as expected
    if values.shape[3] > 1:
        raise ValueError("The T axis must be undefined or a singleton axis")

    # Get the X axis box limits for the quadrilateral coordinates
    x_box_limits = pyferret.get_axis_box_limits(efid, pyferret.ARG2, pyferret.X_AXIS)
    if x_box_limits == None:
        raise ValueError("Unable to determine the X axis box limits")
    lowerxs = x_box_limits[0]
    upperxs = x_box_limits[1]

    # Get the Y axis box limits for the quadrilateral coordinates
    y_box_limits = pyferret.get_axis_box_limits(efid, pyferret.ARG2, pyferret.Y_AXIS)
    if y_box_limits == None:
        raise ValueError("Unable to determine the y axis box limits")
    lowerys = y_box_limits[0]
    upperys = y_box_limits[1]

    # Get the elevation/depth coordinates, or None if they do not exist
    z_coords = pyferret.get_axis_coordinates(efid, pyferret.ARG2, pyferret.Z_AXIS)

    # Create polygons with a single field value
    if z_coords == None:
        sfwriter = shapefile.Writer(shapefile.POLYGON)
    else:
        sfwriter = shapefile.Writer(shapefile.POLYGONZ)
    sfwriter.field(field_name, "N", 20, 7)

    # Write out the shapes and the records
    shape_written = False
    if z_coords == None:
        for j in xrange(len(lowerys)):
            for i in xrange(len(lowerxs)):
                if values[i, j, 0, 0] != missing_value:
                    pyferret.fershp.addquadxyvalues(sfwriter,
                                    ( lowerxs[i], lowerys[j] ),
                                    ( lowerxs[i], upperys[j] ),
                                    ( upperxs[i], upperys[j] ),
                                    ( upperxs[i], lowerys[j] ),
                                    None,
                                    [ float(values[i, j, 0, 0]) ])
                    shape_written = True
    else:
        for k in xrange(len(z_coords)):
            for j in xrange(len(lowerys)):
                for i in xrange(len(lowerxs)):
                    if values[i, j, k, 0] != missing_value:
                        pyferret.fershp.addquadxyvalues(sfwriter,
                                        ( lowerxs[i], lowerys[j] ),
                                        ( lowerxs[i], upperys[j] ),
                                        ( upperxs[i], upperys[j] ),
                                        ( upperxs[i], lowerys[j] ),
                                        z_coords[k],
                                        [ float(values[i, j, k, 0]) ])
                        shape_written = True
    if not shape_written:
        raise ValueError("All values are missing values")
    sfwriter.save(shapefile_name)

    # Create the .prj file from the map projection common name or the WKT description
    pyferret.fershp.createprjfile(map_projection, shapefile_name)
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
