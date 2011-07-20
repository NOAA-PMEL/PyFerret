"""
Helper functions for pyferret shapefile external functions.
"""

import numpy
import shapefile
import os
import os.path
import pyferret.fershapefile.mapprj


def createprjfile(shapefile_mapprj, shapefile_name):
    """
    Creates a map projection (.prj) file for a shapefile.

    Arguments:
        shapefile_mapprj - either the common name or the WKT
                           description of the map projection;
                           if None or blank, "WGS 84" is used.
        shapefile_name   - name of the shapefile; any filename
                           extensions are ignored.
    Raises:
        ValueError if the map projection is invalid.
    """
    # If the string given looks like a WKT description, just use it;
    # otherwise, try to convert the name into a description.
    if (not shapefile_mapprj) or shapefile_mapprj.isspace():
        prj_descript = pyferret.fershapefile.mapprj.name_to_descript("WGS 84")
    elif shapefile_mapprj.startswith('GEOGCS["') or \
         shapefile_mapprj.startswith('PROJCS["'):
        prj_descript = shapefile_mapprj
    else:
        prj_descript = pyferret.fershapefile.mapprj.name_to_descript(shapefile_mapprj)
    (sfname, ext) = os.path.splitext(shapefile_name)
    prjfile = file("%s.prj" % sfname, "w")
    print >>prjfile, prj_descript
    prjfile.close()


def quadxycentroids(xvals, yvals):
    """
    Returns the centroids of X,Y-quadrilaterals whose vertices
    are given by xvals and yvals.

    Arguments:
        xvals - 2D array of X values of the quadrilateral vertices
        yvals - 2D array of Y values of the quadrilateral vertices
        Quadrilaterals are defined by the (xvals, yvals) of
        [i,j] -> [i,j+1] -> [i+1,j+1] -> [i+1,j] -> [i,j]

    Returns:
        Two 2D arrays of X values and Y values of the quadrilateral
        centroids.  The size of each dimension is decreased by one.

    Raises:
        ValueError if the arguments are invalid
    """
    xarray = numpy.asarray(xvals, dtype=float)
    yarray = numpy.asarray(yvals, dtype=float)
    if len(xarray.shape) < 2:
        raise ValueError("xvals and yvals must be (at least) two dimensional")
    if xarray.shape != yarray.shape:
        raise ValueError("xvals and yvals must have the same dimensions")
    sixareas  = xarray[:-1,:-1] * yarray[:-1,1:]  - xarray[:-1,1:]  * yarray[:-1,:-1]
    sixareas += xarray[:-1,1:]  * yarray[1:,1:]   - xarray[1:,1:]   * yarray[:-1,1:]
    sixareas += xarray[1:,1:]   * yarray[1:,:-1]  - xarray[1:,:-1]  * yarray[1:,1:]
    sixareas += xarray[1:,:-1]  * yarray[:-1,:-1] - xarray[:-1,:-1] * yarray[1:,:-1]
    sixareas *= 3.0
    cenxs  = ( xarray[:-1,:-1] * yarray[:-1,1:]  - xarray[:-1,1:]  * yarray[:-1,:-1] ) \
             * ( xarray[:-1,:-1] + xarray[:-1,1:]  )
    cenxs += ( xarray[:-1,1:]  * yarray[1:,1:]   - xarray[1:,1:]   * yarray[:-1,1:] ) \
             * ( xarray[:-1,1:]  + xarray[1:,1:]   )
    cenxs += ( xarray[1:,1:]   * yarray[1:,:-1]  - xarray[1:,:-1]  * yarray[1:,1:] ) \
             * ( xarray[1:,1:]   + xarray[1:,:-1]  )
    cenxs += ( xarray[1:,:-1]  * yarray[:-1,:-1] - xarray[:-1,:-1] * yarray[1:,:-1] ) \
             * ( xarray[1:,:-1]  + xarray[:-1,:-1] )
    cenxs /= sixareas
    cenys  = ( xarray[:-1,:-1] * yarray[:-1,1:]  - xarray[:-1,1:]  * yarray[:-1,:-1] ) \
             * ( yarray[:-1,:-1] + yarray[:-1,1:]  )
    cenys += ( xarray[:-1,1:]  * yarray[1:,1:]   - xarray[1:,1:]   * yarray[:-1,1:] ) \
             * ( yarray[:-1,1:]  + yarray[1:,1:]   )
    cenys += ( xarray[1:,1:]   * yarray[1:,:-1]  - xarray[1:,:-1]  * yarray[1:,1:] ) \
             * ( yarray[1:,1:]   + yarray[1:,:-1]  )
    cenys += ( xarray[1:,:-1]  * yarray[:-1,:-1] - xarray[:-1,:-1] * yarray[1:,:-1] ) \
             * ( yarray[1:,:-1]  + yarray[:-1,:-1] )
    cenys /= sixareas
    return (cenxs, cenys)


def quadxycenters(xvals, yvals):
    """
    Returns the average centers of X,Y-quadrilaterals whose vertices
    are given by xvals and yvals.

    Arguments:
        xvals - 2D array of X values of the quadrilateral vertices
        yvals - 2D array of Y values of the quadrilateral vertices
        Quadrilaterals are defined by the (xvals, yvals) of
        [i,j] -> [i,j+1] -> [i+1,j+1] -> [i+1,j] -> [i,j]

    Returns:
        Two 2D arrays of X values and Y values of the quadrilateral
        average centers.  The size of each dimension is decreased by one.

    Raises:
        ValueError if the arguments are invalid
    """
    xarray = numpy.asarray(xvals, dtype=float)
    yarray = numpy.asarray(yvals, dtype=float)
    if len(xarray.shape) < 2:
        raise ValueError("xvals and yvals must be (at least) two dimensional")
    if xarray.shape != yarray.shape:
        raise ValueError("xvals and yvals must have the same dimensions")
    cenxs  = 0.25 * ( xarray[:-1,:-1] + xarray[:-1,1:] + xarray[1:,1:] + xarray[1:,:-1] )
    cenys  = 0.25 * ( yarray[:-1,:-1] + yarray[:-1,1:] + yarray[1:,1:] + yarray[1:,:-1] )
    return (cenxs, cenys)


def addquadxyvalues(sfwriter, pt0, pt1, pt2, pt3, zcoord, vals):
    """
    Adds a quadrilateral shape to sfwriter defined by the X,Y vertices
    pt0 - pt1 - pt2 - pt3 - pt0, and possibly the common Z coordinate zcoord,
    along with the associated values in vals.

    Arguments:
       sfwriter - the shapefile.Writer object to add the shape and values to
       pt1, pt2,
       pt3, pt4 - the (X,Y) numeric coordinates of the vertices of the simple
                  quadrilateral; in sequence, but not necessarily the correct
                  winding.  Any coordinates after the first two in each point
                  are ignored.
       zcoord   - the numeric Z coordinate for this quadrilateral; may be None
       vals     - the list of values to be associated with this shape.  The
                  fields for these values must already have been created in
                  sfwriter.
    """
    # Get the correct polygon type
    if zcoord != None:
        shapetype = shapefile.POLYGONZ
    else:
        shapetype = shapefile.POLYGON
    x0 = float(pt0[0]); y0 = float(pt0[1])
    x1 = float(pt1[0]); y1 = float(pt1[1])
    x2 = float(pt2[0]); y2 = float(pt2[1])
    x3 = float(pt3[0]); y3 = float(pt3[1])
    # Compute 2 * signed area of this simple quadrilateral
    dqarea  = x0 * y1 - x1 * y0
    dqarea += x1 * y2 - x2 * y1
    dqarea += x2 * y3 - x3 * y2
    dqarea += x3 * y0 - x0 * y3
    # Create the correctly ordered array of coordinates for this single shape part
    part = [ ]
    if dqarea < 0.0:
        # negative means clockwise which is desired direction
        part.append([ x0, y0 ])
        part.append([ x1, y1 ])
        part.append([ x2, y2 ])
        part.append([ x3, y3 ])
        part.append([ x0, y0 ])
    else:
        # positive means counterclockwise so reverse ordering
        part.append([ x0, y0 ])
        part.append([ x3, y3 ])
        part.append([ x2, y2 ])
        part.append([ x1, y1 ])
        part.append([ x0, y0 ])
    # Append the Z coordinate if given
    if zcoord != None:
        z = float(zcoord)
        for pt in part:
            pt.append(z)
    # Add the shape
    sfwriter.poly([ part, ], shapetype)
    # Add the values for this shape
    sfwriter.record(*vals)


#
# The following is only for testing this module from the command line
#
if __name__ == "__main__":

    shapefilename = "testsf"
    wgs84_descript = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
    wgs84upsnorth_descript = 'PROJCS["WGS 84 / UPS North",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",90],PARAMETER["central_meridian",0],PARAMETER["scale_factor",0.994],PARAMETER["false_easting",2000000],PARAMETER["false_northing",2000000],UNIT["metre",1]]'

    # Test createprjfile
    createprjfile(None, "%s.jnk" % shapefilename)
    prjfile = file("%s.prj" % shapefilename, "r")
    datalines = prjfile.readlines()
    prjfile.close()
    if len(datalines) != 1:
        raise ValueError("None for mapprj: more than one line given in the .prj file")
    descript = datalines[0].strip()
    if descript != wgs84_descript:
        raise ValueError("None for mapprj:\n" \
                         "    expect: %s\n" \
                         "    found:  %s" % (wgs84_descript, descript))
    del prjfile, datalines, descript

    prjjunk = 'GEOGCS["Junk",DATUM["Junk"]]'
    createprjfile(prjjunk, "%s.jnk" % shapefilename)
    prjfile = file("%s.prj" % shapefilename, "r")
    datalines = prjfile.readlines()
    prjfile.close()
    if len(datalines) != 1:
        raise ValueError("Junk for mapprj: more than one line given in the .prj file")
    descript = datalines[0].strip()
    if descript != prjjunk:
        raise ValueError("Junk for mapprj:\n" \
                         "    expect: %s\n" \
                         "    found:  %s" % (prjjunk, descript))
    del prjjunk, prjfile, datalines, descript

    createprjfile("WGS 84 / UPS North", "%s.jnk" % shapefilename)
    prjfile = file("%s.prj" % shapefilename, "r")
    datalines = prjfile.readlines()
    prjfile.close()
    if len(datalines) != 1:
        raise ValueError("'WGS 84 / UPS North' for mapprj: more than one line given in the .prj file")
    descript = datalines[0].strip()
    if descript != wgs84upsnorth_descript:
        raise ValueError("'WGS 84 / UPS North' for mapprj:\n" \
                         "    expect: %s\n" \
                         "    found:  %s" % (wgs84upsnorth_descript, descript))
    del prjfile, datalines, descript
    print "createprjfile: SUCCESS"

    # Test quadxycentroids
    xvals = ( ( 0, 1 ), ( 3, 4 ) )
    yvals = ( ( 0, 2 ), ( 1, 3 ) )
    expectx = [ [ 2.0 ] ]
    expecty = [ [ 1.5 ] ]
    (centx, centy) = quadxycentroids(xvals, yvals)
    if not numpy.allclose(centx, expectx):
        raise ValueError("Centroid X values: expected %s; found %s" % \
                          (str(expectx), str(centx)))
    if not numpy.allclose(centy, expecty):
        raise ValueError("Centroid Y values: expected %s; found %s" % \
                          (str(expecty), str(centy)))
    del xvals, yvals, expectx, expecty, centx, centy
    xvals = ( ( 0, 1 ), ( 2, 3 ) )
    yvals = ( ( 0, 2 ), ( 1, 5 ) )
    expectx = [ [ 39.0 / 24.0 ] ]
    expecty = [ [ 49.0 / 24.0 ] ]
    (centx, centy) = quadxycentroids(xvals, yvals)
    if not numpy.allclose(centx, expectx):
        raise ValueError("Centroid X values: expected %s; found %s" % \
                          (str(expectx), str(centx)))
    if not numpy.allclose(centy, expecty):
        raise ValueError("Centroid Y values: expected %s; found %s" % \
                          (str(expecty), str(centy)))
    del xvals, yvals, expectx, expecty, centx, centy
    print "quadxycentroids: SUCCESS"

    # Test quadxycenters
    xvals = ( ( 0, 1 ), ( 3, 4 ) )
    yvals = ( ( 0, 2 ), ( 1, 3 ) )
    expectx = [ [ 2.0 ] ]
    expecty = [ [ 1.5 ] ]
    (centx, centy) = quadxycenters(xvals, yvals)
    if not numpy.allclose(centx, expectx):
        raise ValueError("Centroid X values: expected %s; found %s" % \
                          (str(expectx), str(centx)))
    if not numpy.allclose(centy, expecty):
        raise ValueError("Centroid Y values: expected %s; found %s" % \
                          (str(expecty), str(centy)))
    del xvals, yvals, expectx, expecty, centx, centy
    xvals = ( ( 0, 1 ), ( 2, 3 ) )
    yvals = ( ( 0, 2 ), ( 1, 5 ) )
    expectx = [ [ 1.5 ] ]
    expecty = [ [ 2.0 ] ]
    (centx, centy) = quadxycenters(xvals, yvals)
    if not numpy.allclose(centx, expectx):
        raise ValueError("Centroid X values: expected %s; found %s" % \
                          (str(expectx), str(centx)))
    if not numpy.allclose(centy, expecty):
        raise ValueError("Centroid Y values: expected %s; found %s" % \
                          (str(expecty), str(centy)))
    del xvals, yvals, expectx, expecty, centx, centy
    print "quadxycenters: SUCCESS"

    # Test addquadxyvalues
    zval = -5.34
    coords = [ [0.0, 0.0], [1.0, 0.0], [1.0, -1.0], [2.0, 1.0] ]
    vals = [ 3.28573, 7.46952 ]
    expectedxy = [ coords[0], coords[3], coords[2], coords[1], coords[0] ]
    expectedz = [ zval, zval, zval, zval, zval ]
    # Create the shapefile
    sfwriter = shapefile.Writer(shapefile.POLYGONZ)
    sfwriter.field("Val0", "N", 20, 7)
    sfwriter.field("Val1", "N", 20, 7)
    # Add the shape and values and save the shapefile
    addquadxyvalues(sfwriter, coords[0], coords[1], coords[2], coords[3], zval, vals)
    sfwriter.save(shapefilename)
    del zval, coords, sfwriter
    # Read the shapefile and check the shape and values
    sfreader = shapefile.Reader(shapefilename)
    shapes = sfreader.shapes()
    if len(shapes) != 1:
        raise ValueError("Expected one shape; found %d" % len(shapes))
    if shapes[0].shapeType != shapefile.POLYGONZ:
        raise ValueError("Expected shapetype %d; found %d" % \
                         (shapefile.POLYGONZ, shapes[0].shapeType))
    if not numpy.allclose(shapes[0].points, expectedxy):
        raise ValueError("Expected (X,Y) coordinates %s; found %s" % \
                         (str(expectedxy), str(shapes[0].points)))
    if not numpy.allclose(shapes[0].z, expectedz):
        raise ValueError("Expected Z coordinates %s; found %s" % \
                         (str(expectedz), str(shapes[0].z)))
    records = sfreader.records()
    if len(records) != 1:
        raise ValueError("Expected one set of records; found %d" % len(records))
    if not numpy.allclose(records[0], vals):
        raise ValueError("Expected record values %s; found %s" % \
                         (str(vals), str(records[0])))
    del expectedxy, expectedz, sfreader, shapes, records
    os.remove("%s.dbf" % shapefilename)
    os.remove("%s.shp" % shapefilename)
    os.remove("%s.shx" % shapefilename)
    os.remove("%s.prj" % shapefilename)

    print "addquadxyvalues: SUCCESS"

