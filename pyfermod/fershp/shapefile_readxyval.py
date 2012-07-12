"""
Returns X and Y coordinates (presumably longitude and
latitude), as well as a value for shapes from a shapefile.
The missing value separates coordinates between shapes.
"""

import numpy
import pyferret
import shapefile

def ferret_init(efid):
    """
    Initialization for the shapefile_readxyval PyEF
    """
    retdict = { "numargs": 3,
                "descript": "Returns X, Y, and a value from shapes in a shapefile.  "
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
    Abstract axis limits for the shapefile_readxyval PyEF
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
    return ( (1, maxpts), (1, 3), None, None, None, None, )


def ferret_compute(efid, result, resbdf, inputs, inpbdfs):
    """
    Read the shapefile named in inputs[0] and assign result[:,0,0,0]
    with the X coordinates, result[:,1,0,0] with the Y coordinates
    of the shapes contained in the shapefile.  The missing value,
    resbdf, is assigned as the coordinates of a point separating
    different shapes.  Also assigns result[:,2,0,0] with the value
    of the field named in inputs[1] associated with each shape.
    """
    # Initialize all results to the missing value flag to make it easier later on
    result[:,:,:,:,:,:] = resbdf

    # Open the shapefile for reading and read the metadata
    sf = shapefile.Reader(inputs[0])

    # Find the index of the desired field in the shapefile
    fieldname = inputs[1].strip()
    # No function currently in the shapefile module to do this, so a bit of a hack here
    # Each field in shapefile is a tuple (name, type, size, precision)
    for k in xrange(len(sf.fields)):
        if sf.fields[k][0] == fieldname:
            break
    else:
        print "Known fields (name, type, size, precision):"
        for field in sf.fields:
            if field[0] != 'DeletionFlag':
                print "    %s" % str(field)
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
            for pt in shp.points:
                result[pt_index,:2,0,0,0,0] = pt[:2]
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
        result[rec_index,2,0,0,0,0] = float(rec[field_index])
        rec_index += 1
        # only get field values for shapes that were read
        if rec_index >= num_shapes:
            break


#
# The rest of this is for testing from the command line
#
if __name__ == "__main__":
    import time

    # make sure ferret_init does not cause problems
    info = ferret_init(0)

    resbdf = numpy.array([-9999.0], dtype=numpy.float64)
    inpbdfs = numpy.array([-8888.0, -7777.0], dtype=numpy.float64)
    maxpts = 3200 * 2400
    result = -6666.0 * numpy.ones((maxpts,3,1,1,1,1), dtype=numpy.float64, order='F')
    print "ferret_compute start: time = %s" % time.asctime()
    # INTPTLAT10 == latitude of an internal point in each county
    ferret_compute(0, result, resbdf, ("tl_2010_us_county10", "INTPTLAT10", maxpts, ), inpbdfs)
    print "ferret_compute done; time = %s" % time.asctime()
    good_x = numpy.logical_and((-180.0 <= result[:,0,0,0,0,0]), (result[:,0,0,0,0,0] <= -65.0))
    good_x = numpy.logical_or(good_x,
                 numpy.logical_and((172.0 <= result[:,0,0,0,0,0]), (result[:,0,0,0,0,0] <= 180.0)))
    good_y = numpy.logical_and((17.0 <= result[:,1,0,0,0,0]), (result[:,1,0,0,0,0] <= 72.0))
    if numpy.logical_xor(good_x, good_y).any():
        raise ValueError("good_x != good_y")
    missing_x = ( result[:,0,0,0,0,0] == resbdf )
    if numpy.logical_xor(good_x, numpy.logical_not(missing_x)).any():
        raise ValueError("good_x != not missing_x")
    missing_y = ( result[:,1,0,0,0,0] == resbdf )
    if numpy.logical_xor(good_y, numpy.logical_not(missing_y)).any():
        raise ValueError("good_y != not missing_y")
    count = 0
    at_end = False
    shape_num = 0
    total = 0
    for k in xrange(result.shape[0]):
        if missing_x[k]:
            if count == 0:
                at_end = True
            else:
                # print "Count[%d] = %d" % (shape_num, count)
                shape_num += 1
                total += count + 1
                count = 0
        elif at_end:
            raise ValueError("More than one missing value between shapes")
        else:
            count += 1
    total += count
    good_val = numpy.logical_and((17.0 <= result[:,2,0,0,0,0]), (result[:,2,0,0,0,0] <= 72.0))
    missing_val = ( result[:,2,0,0,0,0] == resbdf )
    if numpy.logical_xor(good_val, numpy.logical_not(missing_val)).any():
        raise ValueError("good_val != not missing_val")
    num_good = len(result[:,2,0,0,0,0][good_val])
    if num_good != shape_num:
        raise ValueError("number of values: expected %d, found %d" % (shape_num, num_good))
    print "shapefile_readxyval: SUCCESS"

