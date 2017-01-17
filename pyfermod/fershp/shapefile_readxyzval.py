"""
Returns X, Y, and Z coordinates (presumably longitude,
latitude, and level), as well as a value for shapes from
a shapefile.  The missing value separates coordinates
between shapes.
"""

from __future__ import print_function

import numpy
import pyferret
import shapefile

def ferret_init(efid):
    """
    Initialization for the shapefile_readxyzval PyEF
    """
    retdict = { "numargs": 3,
                "descript": "Returns X, Y, Z, and a value from shapes in a shapefile.  "
                            "Missing value separates shapes.",
                "restype": pyferret.FLOAT_ARRAY,
                "axes": ( pyferret.AXIS_ABSTRACT,
                          pyferret.AXIS_ABSTRACT,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST, ),
                "argnames": ( "SHAPEFILE", "VALNAME", "MAXPTS", ),
                "argdescripts": ( "Shapefile name (any extension given is ignored)",
                                  "Name of value to retrieve",
                                  "Max. number of points to return (-1 for all, but reads shapefile twice)", ),
                "argtypes": ( pyferret.STRING_ONEVAL,
                              pyferret.STRING_ONEVAL,
                              pyferret.FLOAT_ONEVAL, ),
                "influences": ( (False, False, False, False, False, False),
                                (False, False, False, False, False, False),
                                (False, False, False, False, False, False), ),
              }
    return retdict


def ferret_result_limits(efid):
    """
    Abstract axis limits for the shapefile_readxyzval PyEF
    """
    maxpts = pyferret.get_arg_one_val(efid, pyferret.ARG3)
    maxpts = int(maxpts)
    if maxpts == -1:
        shapefile_name = pyferret.get_arg_one_val(efid, pyferret.ARG1)
        sf = shapefile.Reader(shapefile_name)
        maxpts = 0
        for shp in sf.shapes():
            maxpts += len(shp.points) + 1
    elif maxpts < 1:
        raise ValueError("MAXPTS must be a positive integer or -1")
    return ( (1, maxpts), (1, 4), None, None, None, None, )


def ferret_compute(efid, result, resbdf, inputs, inpbdfs):
    """
    Read the shapefile named in inputs[0] and assign result[:,0,0,0]
    with the X coordinates, result[:,1,0,0] with the Y coordinates,
    and result[:,2,0,0] with the Z coordinates of the shapes contained
    in the shapefile.  The missing value, resbdf, is assigned as the
    coordinates of a point separating different shapes.  Also assigns
    result[:,3,0,0] with the value of the field named in inputs[1]
    associated with each shape.
    """
    # Initialize all results to the missing value flag to make it easier later on
    result[:,:,:,:,:,:] = resbdf

    # Open the shapefile for reading and read the metadata
    sf = shapefile.Reader(inputs[0])

    # Find the index of the desired field in the shapefile
    fieldname = inputs[1].strip()
    # No function currently in the shapefile module to do this, so a bit of a hack here
    # Each field in shapefile is a tuple (name, type, size, precision)
    for k in range(len(sf.fields)):
        if sf.fields[k][0] == fieldname:
            break
    else:
        print("Known fields (name, type, size, precision):")
        for field in sf.fields:
            if field[0] != 'DeletionFlag':
                print("    %s" % str(field))
        raise ValueError("No field with the name '%s' found" % fieldname)
    if sf.fields[0][0] == 'DeletionFlag':
        field_index = k - 1
    else:
        field_index = k

    # Retrieve the coordinates of the shapes
    num_shapes = 0
    try:
        pt_index = 0
        for shp in sf.shapes():
            num_shapes += 1
            for (pt, z) in zip(shp.points, shp.z):
                result[pt_index,:2,0,0,0,0] = pt[:2]
                result[pt_index, 2,0,0,0,0] = z
                pt_index += 1
            # missing value coordinates (already assigned) separating shapes
            pt_index += 1
    except IndexError:
        # hit the maximum number of points
        pass
    if num_shapes < 1:
        raise ValueError("No shapes found")

    # Retrieve the field values
    rec_index = 0
    for rec in sf.records():
        result[rec_index,3,0,0,0,0] = float(rec[field_index])
        rec_index += 1
        # only get field values for shapes that were read
        if rec_index >= num_shapes:
            break


#
# The rest of this is for testing from the command line
#
if __name__ == "__main__":
    # make sure ferret_init does not cause problems
    info = ferret_init(0)

    # this is tested under shapefile_writexyzval
    print("shapefile_readxyzval: SUCCESS (limited)")
    print("    run shapefile_writexyzval for full test")

