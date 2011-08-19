"""
Returns the X, Y, and Z (presumably longitude, latitude, level)
coordinates from the points in the indicated shapefile.
The missing value separates coordinates between shapes.
"""

import numpy
import pyferret
import shapefile

def ferret_init(efid):
    """
    Initialization for the shapefile_readxyz PyEF
    """
    retdict = { "numargs": 2,
                "descript": "Returns X,Y,Z coordinates of shapes from shapefile.  "
                            "Missing value separates shapes.",
                "restype": pyferret.FLOAT_ARRAY,
                "axes": ( pyferret.AXIS_ABSTRACT,
                          pyferret.AXIS_ABSTRACT,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST, ),
                "argnames": ( "SHAPEFILE", "MAXPTS", ),
                "argdescripts": ( "Shapefile name (any extension given is ignored)",
                                  "Max. number of points to return (-1 for all, but reads shapefile twice)", ),
                "argtypes": ( pyferret.STRING_ONEVAL,
                              pyferret.FLOAT_ONEVAL, ),
                "influences": ( (False, False, False, False),
                                (False, False, False, False), ),
              }
    return retdict


def ferret_result_limits(efid):
    """
    Abstract axis limits for the shapefile_readxyz PyEF
    """
    maxpts = pyferret.get_arg_one_val(efid, pyferret.ARG2)
    maxpts = int(maxpts)
    if maxpts == -1:
        shapefile_name = pyferret.get_arg_one_val(efid, pyferret.ARG1)
        sf = shapefile.Reader(shapefile_name)
        maxpts = 0
        for shp in sf.shapes():
            maxpts += len(shp.points) + 1
    elif maxpts < 1:
        raise ValueError("MAXPTS must be a positive integer or -1")
    return ( (1, maxpts), (1, 3), None, None, )


def ferret_compute(efid, result, resbdf, inputs, inpbdfs):
    """
    Read the shapefile named in inputs[0] and assign result[:,0,0,0]
    the X coordinates, result[:,1,0,0] with the Y coordinates, and
    result[:,2,0,0] with the Z coordinates of the shapes contained
    in the shapefile.  The missing value, resbdf, is assigned as the
    coordinates of a point separating different shapes.
    """
    result[:,:,:,:] = resbdf
    sf = shapefile.Reader(inputs[0])
    try:
        pt_index = 0
        for shp in sf.shapes():
            for (pt,z) in zip(shp.points,shp.z):
                result[pt_index,:2,0,0] = pt[:2]
                result[pt_index, 2,0,0] = z
                pt_index += 1
            # missing value coordinates (already assigned) separating shapes
            pt_index += 1
    except IndexError:
        # hit the maximum number of points
        pass


#
# The rest of this is for testing from the command line
#
if __name__ == "__main__":
    # make sure ferret_init does not cause problems
    info = ferret_init(0)

    # this is tested under shapefile_writexyzval
    print "shapefile_readxyz: SUCCESS (limited)"
    print "    run shapefile_writexyzval for full test"

