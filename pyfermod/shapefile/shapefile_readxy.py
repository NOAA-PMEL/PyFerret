"""
Returns the X,Y (presumably longitude,latitude)
coordinates from the points in the indicated shapefile.
The missing value separates coordinates between shapes.
"""

import numpy
import pyferret
import shapefile

def ferret_init(id):
    """
    Initialization for the shapefile_readxy PyEF
    """
    retdict = { "numargs": 2,
                "descript": "Returns X,Y coordinates from the points in the indicated shapefile.  "
                            "The missing value separates coordinates between shapes.",
                "restype": pyferret.FLOAT_ARRAY,
                "axes": ( pyferret.AXIS_ABSTRACT,
                          pyferret.AXIS_ABSTRACT,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST, ),
                "argnames": ( "SHAPEFILE", "MAXPTS", ),
                "argdescripts": ( "Name of the shapefile (any extension given is ignored)",
                                  "Maximum number of points to return (-1 for all, but reads the shapefile twice)", ),
                "argtypes": ( pyferret.STRING_ONEVAL,
                              pyferret.FLOAT_ONEVAL, ),
                "influences": ( (False, False, False, False),
                                (False, False, False, False), ),
              }
    return retdict


def ferret_result_limits(id):
    """
    Abstract axis limits for the shapefile_readxy PyEF
    """
    maxpts = pyferret.get_arg_one_val(id, pyferret.ARG2)
    maxpts = int(maxpts)
    if maxpts == -1:
        shapefile_name = pyferret.get_arg_one_val(id, pyferret.ARG1)
        sf = shapefile.Reader(shapefile_name)
        maxpts = 0
        for shp in sf.shapes():
            maxpts += len(shp.points) + 1
    elif maxpts < 1:
        raise ValueError("MAXPTS must be a positive integer or -1")
    return ( (1, maxpts), (1, 2), None, None, )


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Read the shapefile named in inputs[0] and assign result[:,0,0,0]
    and result[:,1,0,0] with the X and Y coordinates of the shapes
    contained in the shapefile.  The missing value, resbdf, is assigned
    as the coordinates of a point separating different shapes.
    """
    result[:,:,:,:] = resbdf
    sf = shapefile.Reader(inputs[0])
    try:
        pt_index = 0
        for shp in sf.shapes():
            for pt in shp.points:
                result[pt_index,:,0,0] = pt[:2]
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
    import time

    # make sure ferret_init does not cause problems
    info = ferret_init(0)

    resbdf = numpy.array([-9999.0], dtype=numpy.float32)
    inpbdfs = numpy.array([-8888.0, -7777.0], dtype=numpy.float32)
    maxpts = 3200 * 2400
    result = -6666.0 * numpy.ones((maxpts, 2, 1, 1), dtype=numpy.float32, order='F')
    print "ferret_compute start: time = %s" % time.asctime()
    ferret_compute(0, result, resbdf, ("tl_2010_us_county10", maxpts, ), inpbdfs)
    print "ferret_compute done; time = %s" % time.asctime()
    good_x = numpy.logical_and((-180.0 <= result[:,0,0,0]), (result[:,0,0,0] <= -65.0))
    good_x = numpy.logical_or(good_x,
                 numpy.logical_and((172.0 <= result[:,0,0,0]), (result[:,0,0,0] <= 180.0)))
    good_y = numpy.logical_and((17.0 <= result[:,1,0,0]), (result[:,1,0,0] <= 72.0))
    if numpy.logical_xor(good_x, good_y).any():
        raise ValueError("good_x != good_y")
    missing_x = ( result[:,0,0,0] == resbdf )
    if numpy.logical_xor(good_x, numpy.logical_not(missing_x)).any():
        raise ValueError("good_x != not missing_x")
    missing_y = ( result[:,1,0,0] == resbdf )
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
    print "total (including missing-value separators) = %d" % total
    print "out of a maximum of %d" %  result.shape[0]
    print "number of shapes = %d" % shape_num
    print "SUCCESS"

