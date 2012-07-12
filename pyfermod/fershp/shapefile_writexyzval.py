"""
Creates a shapefile with a given root name using data from given
X, Y, Z, and value arrays (curvilinear-type data).  The shapes
are quadrilaterals derived from the X and Y arrays which can have
varying Z coordinates; i.e., vertices are the (i,j), (i,j+1),
(i+1,j+1), and (i+1,j) elements of the X, Y, and Z coordinates.
The value associated with each quadrilateral comes from the value
array; i.e., the (i,j) element of the value for the previously
mentioned quadrilateral.  Quadrilaterals associated with missing
values are omitted from the shapefile.
"""

import shapefile
import pyferret
import pyferret.fershp

def ferret_init(efid):
    """
    Initialization for the shapefile_writexyzval PyEF
    """
    retdict = { "numargs": 7,
                "descript": "Writes a shapefile of XY quadrilaterals with Z coordinates from the curvilinear data arrays.",
                "restype": pyferret.FLOAT_ARRAY,
                "axes": ( pyferret.AXIS_ABSTRACT,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST, ),
                "argnames": ( "SHAPEFILE", "GRIDX", "GRIDY", "GRIDZ", "VALUE", "VALNAME", "MAPPRJ"),
                "argdescripts": ( "Shapefile name (any extension given is ignored)",
                                  "X values (longitudes) for the quad. grid; must be 2D on X and Y axes",
                                  "Y values (latitudes) for the quad. grid; must be 2D on X and Y axes",
                                  "Z values (levels) for the quad. grid; must be 2D on X and Y axes",
                                  "Shape values; must be 2D on X and Y axes",
                                  "Name for the shape value",
                                  "Common name or WKT description of map projection; " \
                                      "if blank, WGS 84 is used", ),
                "argtypes": ( pyferret.STRING_ONEVAL,
                              pyferret.FLOAT_ARRAY,
                              pyferret.FLOAT_ARRAY,
                              pyferret.FLOAT_ARRAY,
                              pyferret.FLOAT_ARRAY,
                              pyferret.STRING_ONEVAL,
                              pyferret.STRING_ONEVAL, ),
                "influences": ( (False, False, False, False, False, False),
                                (False, False, False, False, False, False),
                                (False, False, False, False, False, False),
                                (False, False, False, False, False, False),
                                (False, False, False, False, False, False),
                                (False, False, False, False, False, False),
                                (False, False, False, False, False, False), ),
              }
    return retdict


def ferret_result_limits(efid):
    """
    Abstract axis limits for the shapefile_writexyzval PyEF
    """
    return ( (1, 1), None, None, None, None, None, )


def ferret_compute(efid, result, resbdf, inputs, inpbdfs):
    """
    Create the shapefile named in inputs[0] using the grid X coordinates
    given in inputs[1], grid Y coordinates given in inputs[2], grid Z
    coordinates given in inputs[3], and shape values given in inputs[4].
    The X, Y, and Z coordinates are used for the quadrilaterals vertices
    and must have an additional value along each dimension.  The value
    [i, j] is used for the quadrilateral with diagonal corners [i, j] and
    [i+1, j+1].  Quadrilateral associated with missing values are omitted
    from the shapefile.  The field name for the value in the shapefile
    given in inputs[5].  Either a common name or a WKT description of the
    map projection for the coordinates should be given in inputs[6].  If
    blank, WGS 84 is used.  If successful, fills result (which might as
    well be a 1x1x1x1 array) with zeros.  If a problem occurs, an error
    will be raised.
    """
    shapefile_name = inputs[0]
    grid_xs = inputs[1]
    grid_ys = inputs[2]
    grid_zs = inputs[3]
    grid_vals = inputs[4]
    missing_val = inpbdfs[4]
    field_name = inputs[5].strip()
    if not field_name:
        field_name = "VALUE"
    map_projection = inputs[6]

    # Verify the shapes are as expected
    if (grid_vals.shape[2] != 1) or (grid_vals.shape[3] != 1) or \
       (grid_vals.shape[4] != 1) or (grid_vals.shape[5] != 1):
        raise ValueError("The Z, T, E, and F axes of VALUE must be undefined or singleton axes")
    exp_shape = ( grid_vals.shape[0] + 1, grid_vals.shape[1] + 1, 1, 1, 1, 1 )
    if (grid_xs.shape != exp_shape) or \
       (grid_ys.shape != exp_shape) or \
       (grid_zs.shape != exp_shape):
         raise ValueError('GRIDX, GRIDY, and GRIDZ must have one more value ' \
                          'along both X and Y axes compared to VALUE')

    # Create polygons with a single field value
    sfwriter = shapefile.Writer(shapefile.POLYGONZ)
    sfwriter.field(field_name, "N", 20, 7)

    # Add the shapes with their values
    shape_written = False
    for j in xrange(grid_vals.shape[1]):
        for i in xrange(grid_vals.shape[0]):
            if grid_vals[i, j, 0, 0, 0, 0] != missing_val:
                shape_written = True
                pyferret.fershp.addquadxyvalues(sfwriter,
                         (grid_xs[i,   j,   0, 0, 0, 0], grid_ys[i,   j,   0, 0, 0, 0]),
                         (grid_xs[i,   j+1, 0, 0, 0, 0], grid_ys[i,   j+1, 0, 0, 0, 0]),
                         (grid_xs[i+1, j+1, 0, 0, 0, 0], grid_ys[i+1, j+1, 0, 0, 0, 0]),
                         (grid_xs[i+1, j,   0, 0, 0, 0], grid_ys[i+1, j,   0, 0, 0, 0]),
                         ( grid_zs[i,   j,   0, 0, 0, 0],
                           grid_zs[i,   j+1, 0, 0, 0, 0],
                           grid_zs[i+1, j+1, 0, 0, 0, 0],
                           grid_zs[i+1, j,   0, 0, 0, 0] ),
                         [ float(grid_vals[i, j, 0, 0, 0, 0]) ])
    if not shape_written:
        raise ValueError("All values are missing values")
    sfwriter.save(shapefile_name)

    # Create the .prj file from the map projection common name or the WKT description
    pyferret.fershp.createprjfile(map_projection, shapefile_name)
    result[:, :, :, :, :, :] = 0


#
# The following is only for testing this module from the command line
#
if __name__ == "__main__":
    import numpy
    import numpy.random
    import os

    shapefilename = "tripolarwz"
    fieldname = "AREA"
    wgs84_descript = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'

    # real world longitudes and latitudes of tripolar coordinates X=80W:60E:10 + 100E:120W:10,Y=45N:85N:5
    geolon_c = numpy.array([
        [  -92.1, -87.7, -82.4, -75.7, -66.7, -53.8, -34.9, -10.0,  14.9,  33.8,
            46.7,  55.7,  62.4,  67.7,  72.1,  87.9,  92.3,  97.6, 104.3, 113.3,
           126.2, 145.1, 170.0, 194.9, 213.8, 226.7, 235.7, 242.4, 247.7, 252.1, 267.9, ],
        [  -86.0, -78.5, -70.2, -60.9, -50.2, -38.1, -24.5, -10.0,   4.5,  18.1,
            30.2,  40.9,  50.2,  58.5,  66.0,  94.0, 101.5, 109.8, 119.1, 129.8,
           141.9, 155.5, 170.0, 184.5, 198.1, 210.2, 220.9, 230.2, 238.5, 246.0, 274.0, ],
        [  -82.3, -73.1, -63.6, -53.7, -43.3, -32.5, -21.4, -10.0,   1.4,  12.5,
            23.3,  33.7,  43.6,  53.1,  62.3,  97.7, 106.9, 116.4, 126.3, 136.7,
           147.5, 158.6, 170.0, 181.4, 192.5, 203.3, 213.7, 223.6, 233.1, 242.3, 277.7, ],
        [  -80.5, -70.6, -60.7, -50.7, -40.6, -30.5, -20.3, -10.0,   0.3,  10.5,
            20.6,  30.7,  40.7,  50.6,  60.5,  99.5, 109.4, 119.3, 129.3, 139.4,
           149.5, 159.7, 170.0, 180.3, 190.5, 200.6, 210.7, 220.7, 230.6, 240.5, 279.5, ],
        [  -80.0, -70.0, -60.0, -50.0, -40.0, -30.0, -20.0, -10.0,   0.0,  10.0,
            20.0,  30.0,  40.0,  50.0,  60.0, 100.0, 110.0, 120.0, 130.0, 140.0,
           150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 280.0, ],
        [  -80.0, -70.0, -60.0, -50.0, -40.0, -30.0, -20.0, -10.0,   0.0,  10.0,
            20.0,  30.0,  40.0,  50.0,  60.0, 100.0, 110.0, 120.0, 130.0, 140.0,
           150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 280.0, ],
        [  -80.0, -70.0, -60.0, -50.0, -40.0, -30.0, -20.0, -10.0,   0.0,  10.0,
            20.0,  30.0,  40.0,  50.0,  60.0,  100.0, 110.0, 120.0, 130.0, 140.0,
           150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 280.0, ],
        [  -80.0, -70.0, -60.0, -50.0, -40.0, -30.0, -20.0, -10.0,   0.0,  10.0,
            20.0,  30.0,  40.0,  50.0,  60.0, 100.0, 110.0, 120.0, 130.0, 140.0,
           150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 280.0, ],
        [  -80.0, -70.0, -60.0, -50.0, -40.0, -30.0, -20.0, -10.0,   0.0,  10.0,
            20.0,  30.0,  40.0,  50.0,  60.0, 100.0, 110.0, 120.0, 130.0, 140.0,
           150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 280.0, ],
    ], dtype=numpy.float64)
    geolon_c = geolon_c.T[:, :, numpy.newaxis, numpy.newaxis, numpy.newaxis, numpy.newaxis]
    geolat_c = numpy.array([
        [ 71.85, 74.69, 77.25, 79.54, 81.58, 83.30, 84.53, 85.00, 84.53, 83.30,
          81.58, 79.54, 77.25, 74.69, 71.85, 71.85, 74.69, 77.25, 79.54, 81.58,
          83.30, 84.53, 85.00, 84.53, 83.30, 81.58, 79.54, 77.25, 74.69, 71.85, 71.85, ],
        [ 70.51, 72.81, 74.83, 76.56, 77.99, 79.08, 79.76, 80.00, 79.76, 79.08,
          77.99, 76.56, 74.83, 72.81, 70.51, 70.51, 72.81, 74.83, 76.56, 77.99,
          79.08, 79.76, 80.00, 79.76, 79.08, 77.99, 76.56, 74.83, 72.81, 70.51, 70.51, ],
        [ 68.71, 70.29, 71.67, 72.83, 73.76, 74.44, 74.86, 75.00, 74.86, 74.44,
          73.76, 72.83, 71.67, 70.29, 68.71, 68.71, 70.29, 71.67, 72.83, 73.76,
          74.44, 74.86, 75.00, 74.86, 74.44, 73.76, 72.83, 71.67, 70.29, 68.71, 68.71, ],
        [ 66.80, 67.60, 68.30, 68.90, 69.37, 69.72, 69.93, 70.00, 69.93, 69.72,
          69.37, 68.90, 68.30, 67.60, 66.80, 66.80, 67.60, 68.30, 68.90, 69.37,
          69.72, 69.93, 70.00, 69.93, 69.72, 69.37, 68.90, 68.30, 67.60, 66.80, 66.80, ],
        [ 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00,
          65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00,
          65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, ],
        [ 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00,
          60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00,
          60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, ],
        [ 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00,
          55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00,
          55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, ],
        [ 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00,
          50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00,
          50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, ],
        [ 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00,
          45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00,
          45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, ],
    ], dtype=numpy.float64)
    geolat_c = geolat_c.T[:, :, numpy.newaxis, numpy.newaxis, numpy.newaxis, numpy.newaxis]

    # Just create random elevations in [0.0,100.0)
    levels = 100.0 * numpy.random.rand(*(geolon_c.shape))
    levels = numpy.array(levels, dtype=numpy.float64, order='F')

    # Make the value an approximate sphere surface area (in square degrees) of the quadrilateral
    vals  = geolon_c[:-1, :-1] * geolat_c[:-1,  1:]
    vals -= geolon_c[:-1,  1:] * geolat_c[:-1, :-1]
    vals += geolon_c[:-1,  1:] * geolat_c[ 1:,  1:]
    vals -= geolon_c[ 1:,  1:] * geolat_c[:-1,  1:]
    vals += geolon_c[ 1:,  1:] * geolat_c[ 1:, :-1]
    vals -= geolon_c[ 1:, :-1] * geolat_c[ 1:,  1:]
    vals += geolon_c[ 1:, :-1] * geolat_c[:-1, :-1]
    vals -= geolon_c[:-1, :-1] * geolat_c[ 1:, :-1]
    vals = 0.5 * numpy.fabs(vals)
    vals *= numpy.cos( 0.25 * numpy.deg2rad(geolat_c[:-1, :-1] + \
                                            geolat_c[:-1,  1:] + \
                                            geolat_c[ 1:,  1:] + \
                                            geolat_c[ 1:, :-1]) )

    # make sure these calls do not generate errors
    info = ferret_init(0)
    del info
    limits = ferret_result_limits(0)
    del limits

    # check that ferret_compute does not cause any errors
    resbdf = numpy.array([-99999.0], dtype=numpy.float64)
    inpbdfs = numpy.array([-88888.0, -77777.0, -66666.0, -55555.0, -44444.0, -33333.0, -22222.0], dtype=numpy.float64)
    result = numpy.ones((1,1,1,1,1,1), dtype=numpy.float64)
    ferret_compute(0, result, resbdf, (shapefilename, geolon_c, geolat_c, levels, vals, fieldname, ""), inpbdfs)
    if result[0,0,0,0,0,0] != 0.0:
        raise ValueError("ferret_compute result array value: expected 0.0, found %f" % result[0,0,0,0,0,0])

    # create the expected arrays returned from shapefile.Reader
    # as well as from shapefile_readxyz and shapefile_readxyzval
    exppoints = []
    expzs = []
    expvals = []
    expcurvals = [ [], [], [], [] ]
    for j in xrange(vals.shape[1]):
        for i in xrange(vals.shape[0]):
            # add the values expected to be returned from shapefile.Reader
            exppoints.append( numpy.array([ [ geolon_c[i,   j,   0, 0, 0, 0], 
                                              geolat_c[i,   j,   0, 0, 0, 0] ],
                                            [ geolon_c[i+1, j,   0, 0, 0, 0], 
                                              geolat_c[i+1, j,   0, 0, 0, 0] ],
                                            [ geolon_c[i+1, j+1, 0, 0, 0, 0], 
                                              geolat_c[i+1, j+1, 0, 0, 0, 0] ],
                                            [ geolon_c[i,   j+1, 0, 0, 0, 0], 
                                              geolat_c[i,   j+1, 0, 0, 0, 0] ],
                                            [ geolon_c[i,   j,   0, 0, 0, 0], 
                                              geolat_c[i,   j,   0, 0, 0, 0] ] ]) )
            expzs.append( numpy.array([ levels[i,   j,   0, 0, 0, 0],
                                        levels[i+1, j,   0, 0, 0, 0],
                                        levels[i+1, j+1, 0, 0, 0, 0],
                                        levels[i,   j+1, 0, 0, 0, 0],
                                        levels[i,   j,   0, 0, 0, 0] ]) )
            expvals.append(vals[i, j, 0, 0, 0, 0])
            # add the expected values returned from shapefile_readxyz and shapefile_readxyzval
            expcurvals[0].extend(exppoints[-1][:,0])
            expcurvals[0].append(resbdf[0])
            expcurvals[1].extend(exppoints[-1][:,1])
            expcurvals[1].append(resbdf[0])
            expcurvals[2].extend(expzs[-1])
            expcurvals[2].append(resbdf[0])
            expcurvals[3].append(expvals[-1])
    # shapefile_readxyz and shapefile_readxyzval return numpy.float64 arrays
    expcurvals[0] = numpy.array(expcurvals[0], dtype=numpy.float64)
    expcurvals[1] = numpy.array(expcurvals[1], dtype=numpy.float64)
    expcurvals[2] = numpy.array(expcurvals[2], dtype=numpy.float64)
    expcurvals[3] = numpy.array(expcurvals[3], dtype=numpy.float64)

    # check the values read using shapefile.Reader
    sfreader = shapefile.Reader(shapefilename)
    shapes = sfreader.shapes()
    records = sfreader.records()
    explen = vals.shape[0] * vals.shape[1]
    if len(shapes) != explen:
        raise ValueError("Expected %d shapes; found %d" % (explen, len(shapes)))
    if len(records) != explen:
        raise ValueError("Expected %d records; found %d" % (explen, len(records)))
    # this does not assume any order that the shapes were written
    for (shape, record) in zip(shapes, records):
        for k in range(len(exppoints)):
            if numpy.allclose(shape.points, exppoints[k], rtol=1.0E-4):
                break
        else:
            raise ValueError("Unexpected X,Y vertices %s" % str(shape.points))
        if not numpy.allclose(shape.z, expzs[k], rtol=1.0E-4, atol=1.0E-5):
            raise ValueError("Expected Zs %s; found %s for shape.points %s" % \
                             (str(expzs[k]), str(shape.z), str(shape.points)))
        if not numpy.allclose(record, expvals[k], rtol=1.0E-4):
            raise ValueError("Expected value %s; found %s for shape.points %s" % \
                             (str(expvals[k]), str(record), str(shape.points)))
        junk = exppoints.pop(k)
        junk = expzs.pop(k)
        junk = expvals.pop(k)
    prjfile = file("%s.prj" % shapefilename, "r")
    datalines = prjfile.readlines()
    prjfile.close()
    if len(datalines) != 1:
        raise ValueError("Number of lines in the .prj file: expected: 1, found %d" % len(datalines))
    descript = datalines[0].strip()
    if descript != wgs84_descript:
        raise ValueError("Description in the .prj file:\n" \
                         "    expect: %s\n" \
                         "    found:  %s" % (wgs84_descript, descript))

    print "shapefile_writexyzval: SUCCESS"

    # Check the result for calling ferret_compute of shapefile_readxyz
    # in this directory.  This assumes the ordering of the shapes does
    # not change, which appears to be the case but is not required.
    import shapefile_readxyz
    maxpts = len(expcurvals[0])
    result = -11111.0 * numpy.ones((maxpts, 3, 1, 1, 1, 1), dtype=numpy.float64, order='F')
    shapefile_readxyz.ferret_compute(0, result, resbdf, (shapefilename, maxpts), inpbdfs[:2])
    if not numpy.allclose(result[:,0,0,0,0,0], expcurvals[0], rtol=1.0E-4):
        raise ValueError("Xs from shapefile_readxyz:\n   expected\n%s\n   found\n%s" % \
                         (str(expcurvals[0]), str(result[:,0,0,0,0,0])))
    if not numpy.allclose(result[:,1,0,0,0,0], expcurvals[1], rtol=1.0E-4):
        raise ValueError("Ys from shapefile_readxyz:\n   expected\n%s\n   found\n%s" % \
                         (str(expcurvals[1]), str(result[:,1,0,0,0,0])))
    if not numpy.allclose(result[:,2,0,0,0,0], expcurvals[2], rtol=1.0E-4, atol=1.0E-5):
        raise ValueError("Zs from shapefile_readxyz:\n   expected\n%s\n   found\n%s" % \
                         (str(expcurvals[2]), str(result[:,2,0,0,0,0])))

    print "shapefile_readxyz: SUCCESS"

    # Check the result for calling ferret_compute of shapefile_readxyzval
    # in this directory.  This assumes the ordering of the shapes does
    # not change, which appears to be the case but is not required.
    import shapefile_readxyzval
    result = -11111.0 * numpy.ones((maxpts, 4, 1, 1, 1, 1), dtype=numpy.float64, order='F')
    shapefile_readxyzval.ferret_compute(0, result, resbdf, (shapefilename, fieldname, maxpts), inpbdfs[:3])
    if not numpy.allclose(result[:,0,0,0,0,0], expcurvals[0], rtol=1.0E-4):
        raise ValueError("Xs from shapefile_readxyzval:\n   expected\n%s\n   found\n%s" % \
                         (str(expcurvals[0]), str(result[:,0,0,0,0,0])))
    if not numpy.allclose(result[:,1,0,0,0,0], expcurvals[1], rtol=1.0E-4):
        raise ValueError("Ys from shapefile_readxyzval:\n   expected\n%s\n   found\n%s" % \
                         (str(expcurvals[1]), str(result[:,1,0,0,0,0])))
    if not numpy.allclose(result[:,2,0,0,0,0], expcurvals[2], rtol=1.0E-4, atol=1.0E-5):
        raise ValueError("Zs from shapefile_readxyzval:\n   expected\n%s\n   found\n%s" % \
                         (str(expcurvals[2]), str(result[:,2,0,0,0,0])))
    numvals = len(expcurvals[3])
    if not numpy.allclose(result[:numvals,3,0,0,0,0], expcurvals[3], rtol=1.0E-4):
        raise ValueError("Values from shapefile_readxyzval:\n   expected\n%s\n   found\n%s" % \
                         (str(expcurvals[3]), str(result[:numvals,3,0,0,0,0])))
    if not numpy.allclose(result[numvals:,3,0,0,0,0], resbdf, rtol=1.0E-4):
        raise ValueError("Extra values from shapefile_readxyzval: expected all %s\n   found\n%s" % \
                         (str(float(resbdf[0])), str(result[numvals:,3,0,0,0,0])))

    print "shapefile_readxyzval: SUCCESS"

    os.remove("%s.dbf" % shapefilename)
    os.remove("%s.shp" % shapefilename)
    os.remove("%s.shx" % shapefilename)
    os.remove("%s.prj" % shapefilename)


