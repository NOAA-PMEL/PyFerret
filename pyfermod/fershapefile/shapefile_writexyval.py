"""
Creates a shapefile with a given root name using data from given X, Y,
and value arrays (curvilinear-type data).  The shapes are quadrilaterals
in the X,Y-plane derived from the X and Y arrays.  The value associated
with each shape comes from the value array.
"""

import shapefile
import pyferret
import pyferret.fershapefile

def ferret_init(efid):
    """
    Initialization for the shapefile_writexyval PyEF
    """
    retdict = { "numargs": 6,
                "descript": "Writes a shapefile of XY quadrilaterals from the curvilinear data arrays.",
                "restype": pyferret.FLOAT_ARRAY,
                "axes": ( pyferret.AXIS_ABSTRACT,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST,
                          pyferret.AXIS_DOES_NOT_EXIST, ),
                "argnames": ( "SHAPEFILE", "GRIDX", "GRIDY", "GRIDVALS", "GRIDLOC", "MAPPRJ"),
                "argdescripts": ( "Name for the shapefile (any extension given is ignored)",
                                  "X (Longitude) grid values for the shapefile; must be 2D on X and Y",
                                  "Y (Latitude) grid values for the shapefile; must be 2D on X and Y",
                                  "Values for the shapes; must be 2D on X and Y",
                                  'Location of the X and Y grid values, either "centroid", "center" or "corner"',
                                  "Either a common name or a WKT description for the map projection; " \
                                      "if blank, WGS 84 is used", ),
                "argtypes": ( pyferret.STRING_ONEVAL,
                              pyferret.FLOAT_ARRAY,
                              pyferret.FLOAT_ARRAY,
                              pyferret.FLOAT_ARRAY,
                              pyferret.STRING_ONEVAL,
                              pyferret.STRING_ONEVAL, ),
                "influences": ( (False, False, False, False),
                                (False, False, False, False),
                                (False, False, False, False),
                                (False, False, False, False),
                                (False, False, False, False),
                                (False, False, False, False), ),
              }
    return retdict


def ferret_result_limits(efid):
    """
    Abstract axis limits for the shapefile_writexyval PyEF
    """
    return ( (1, 1), None, None, None, )


def ferret_compute(efid, result, resbdf, inputs, inpbdfs):
    """
    Create the shapefile named in inputs[0] using the grid X coordinates given
    in inputs[1], grid Y coordinates given in inputs[2], and grid values given
    in inputs[3].  If inputs[4] is "center", then the values are taken to be
    at the X,Y coordinates and quadrilaterals are created from bounding boxes
    around these points.  If inputs[4] is "corner", the X,Y coordinates are
    used for the quadrilaterals and must have an additional value along each
    dimension, and the value [i,j] is used for the quadrilateral with diagonal
    corners [i, j] and [i+1, j+1].  Either a common name or a WKT description
    of the map projection for the coordinated should be given in inputs[5].
    If blank, WGS 84 is used.  If successful, fills result (which might as
    well be a 1x1x1x1 array) with zeros.  If problems, an error will be raised.
    """
    shapefile_name = inputs[0]
    grid_xs = inputs[1]
    grid_ys = inputs[2]
    grid_vals = inputs[3]
    grid_loc = inputs[4].lower()
    shapefile_mapprj = inputs[5]
    # Verify the shapes are as expected
    if (grid_vals.shape[2] != 1) or (grid_vals.shape[3] != 1):
        raise ValueError("GRIDVALS Z and T axes must be undefined or singleton axes")
    if grid_loc == "center":
        if (grid_xs.shape != grid_vals.shape) or (grid_ys.shape != grid_vals.shape):
            raise ValueError('For a "center" grid, GRIDX and GRIDY must have ' \
                             'the same dimensions as GRIDVALS')
    elif grid_loc == "corner":
        exp_shape = ( grid_vals.shape[0] + 1, grid_vals.shape[1] + 1, 1, 1 )
        if (grid_xs.shape != exp_shape) or (grid_ys.shape != exp_shape):
            raise ValueError('For a "corner" grid, GRIDX and GRIDY must have ' \
                             'one additional element along the X and Y axes')
    else:
        raise ValueError("Unknown GRIDLOCATION of %s" % grid_loc)
    # Create the shapefile writer object
    sfwriter = shapefile.Writer(shapefile.POLYGON)
    # Create the field for the value
    # TODO: get reasonable name and sizes for the field
    sfwriter.field("Value", "N", 20, 7)
    # Add all the shapes with their values
    if grid_loc == "center":
         # TODO: stubbed
         raise ValueError("Stubbed")
    elif grid_loc == "corner":
        for j in xrange(grid_vals.shape[1]):
            for i in xrange(grid_vals.shape[0]):
                pyferret.fershapefile.addquadxyvalues(sfwriter,
                         (grid_xs[i,   j,   0, 0], grid_ys[i,   j,   0, 0]),
                         (grid_xs[i,   j+1, 0, 0], grid_ys[i,   j+1, 0, 0]),
                         (grid_xs[i+1, j+1, 0, 0], grid_ys[i+1, j+1, 0, 0]),
                         (grid_xs[i+1, j,   0, 0], grid_ys[i+1, j,   0, 0]),
                         None, grid_vals[i, j, 0, :].tolist())
    else:
        raise ValueError("Unknown GRIDLOCATION of %s" % grid_loc)
    sfwriter.save(shapefile_name)
    # Create the .prj file from the map projection common name or the WKT description
    pyferret.fershapefile.createprjfile(shapefile_mapprj, shapefile_name)
    result[:, :, :, :] = 0


#
# The following is only for testing this module from the command line
#
if __name__ == "__main__":
    import numpy
    import os

    shapefilename = "testsf"
    wgs84_descript = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'

    # values from tripolar coordinates X=90E:120W:10,Y=45N:85N:5
    geolon_c = numpy.array([
     [ 83.8,  87.9,  92.3,  97.6, 104.3, 113.3, 126.2, 145.1, 170.0, 194.9, 213.8, 226.7, 235.7, 242.4, 247.7, 252.1 ],
     [ 86.9,  94.0, 101.5, 109.8, 119.1, 129.8, 141.9, 155.5, 170.0, 184.5, 198.1, 210.2, 220.9, 230.2, 238.5, 246.0 ],
     [ 88.8,  97.7, 106.9, 116.4, 126.3, 136.7, 147.5, 158.6, 170.0, 181.4, 192.5, 203.3, 213.7, 223.6, 233.1, 242.3 ],
     [ 89.7,  99.5, 109.4, 119.3, 129.3, 139.4, 149.5, 159.7, 170.0, 180.3, 190.5, 200.6, 210.7, 220.7, 230.6, 240.5 ],
     [ 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0 ],
     [ 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0 ],
     [ 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0 ],
     [ 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0 ],
     [ 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0 ],
    ], dtype=numpy.float32)
    geolon_c = geolon_c.T[:, :, numpy.newaxis, numpy.newaxis]
    geolat_c = numpy.array([
     [ 68.65, 71.85, 74.69, 77.25, 79.54, 81.58, 83.30, 84.53, 85.00, 84.53, 83.30, 81.58, 79.54, 77.25, 74.69, 71.85 ],
     [ 67.92, 70.51, 72.81, 74.83, 76.56, 77.99, 79.08, 79.76, 80.00, 79.76, 79.08, 77.99, 76.56, 74.83, 72.81, 70.51 ],
     [ 66.94, 68.71, 70.29, 71.67, 72.83, 73.76, 74.44, 74.86, 75.00, 74.86, 74.44, 73.76, 72.83, 71.67, 70.29, 68.71 ],
     [ 65.93, 66.80, 67.60, 68.30, 68.90, 69.37, 69.72, 69.93, 70.00, 69.93, 69.72, 69.37, 68.90, 68.30, 67.60, 66.80 ],
     [ 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00 ],
     [ 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00, 60.00 ],
     [ 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00, 55.00 ],
     [ 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00 ],
     [ 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00, 45.00 ],
    ], dtype=numpy.float32)
    geolat_c = geolat_c.T[:, :, numpy.newaxis, numpy.newaxis]
    vals  = (geolon_c[:-1, :-1, :, :] * geolat_c[1:,  :-1, :, :] - geolon_c[1:,  :-1, :, :]) * geolat_c[:-1, :-1, :, :]
    vals += (geolon_c[1:,  :-1, :, :] * geolat_c[1:,  1:,  :, :] - geolon_c[1:,  1:,  :, :]) * geolat_c[1:,  :-1, :, :]
    vals += (geolon_c[1:,  1:,  :, :] * geolat_c[:-1, 1:,  :, :] - geolon_c[:-1, 1:,  :, :]) * geolat_c[1:,  1:,  :, :]
    vals += (geolon_c[:-1, 1:,  :, :] * geolat_c[:-1, :-1, :, :] - geolon_c[:-1, :-1, :, :]) * geolat_c[:-1, 1:,  :, :]
    vals *= numpy.cos((geolat_c[:-1, :-1, :, :] + geolat_c[:-1, 1:,  :, :] + \
                       geolat_c[1:,  1:,  :, :] + geolat_c[1:,  :-1, :, :]) * 0.25 * numpy.pi / 180.0)
    vals *= ( numpy.pi / 180.0 )**2

    # make sure these calls do not generate errors
    info = ferret_init(0)
    del info
    limits = ferret_result_limits(0)
    del limits

    resbdf = numpy.array([9999.0], dtype=numpy.float32)
    inpbdfs = numpy.array([8888.0, 7777.0, 6666.0, 5555.0, 4444.0, 3333.0], dtype=numpy.float32)
    result = numpy.ones((1,1,1,1), dtype=numpy.float32)
    ferret_compute(0, result, resbdf, (shapefilename, geolon_c, geolat_c, vals, "corner", ""), inpbdfs)

    sfreader = shapefile.Reader(shapefilename)
    shapes = sfreader.shapes()
    records = sfreader.records()
    explen = vals.shape[0] * vals.shape[1]
    if len(shapes) != explen:
        raise ValueError("Expected %d shapes; found %d" % (explen, len(shapes)))
    if len(records) != explen:
        raise ValueError("Expected %d records; found %d" % (explen, len(records)))
    exppoints = []
    expvals = []
    for j in xrange(vals.shape[1]):
        for i in xrange(vals.shape[0]):
            exppoints.append( numpy.array([ [ geolon_c[i,   j,   0, 0], geolat_c[i,   j,   0, 0] ],
                                            [ geolon_c[i+1, j,   0, 0], geolat_c[i+1, j,   0, 0] ],
                                            [ geolon_c[i+1, j+1, 0, 0], geolat_c[i+1, j+1, 0, 0] ],
                                            [ geolon_c[i,   j+1, 0, 0], geolat_c[i,   j+1, 0, 0] ],
                                            [ geolon_c[i,   j,   0, 0], geolat_c[i,   j,   0, 0] ] ]) )
            expvals.append(vals[i, j, 0, 0])
    for (shape, record) in zip(shapes, records):
        for k in range(len(exppoints)):
            if numpy.allclose(shape.points, exppoints[k], rtol=1.0E-4):
                break
        else:
            raise ValueError("Unexpected vertices %s" % str(shape.points))
        if not numpy.allclose(record, expvals[k], rtol=1.0E-4):
            raise ValueError("Expected value %s; found %s for shape.points %s" % \
                             (str(expvals[k]), str(record), str(shape.points)))
        junk = exppoints.pop(k)
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

    os.remove("%s.dbf" % shapefilename)
    os.remove("%s.shp" % shapefilename)
    os.remove("%s.shx" % shapefilename)
    os.remove("%s.prj" % shapefilename)

    print "shapefile_writexyval: SUCCESS"

