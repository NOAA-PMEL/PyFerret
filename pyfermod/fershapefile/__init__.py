"""
Helper functions for pyferret shapefile external functions.
"""

import shapefile

def addquadxyvalues(sfwriter, pt0, pt1, pt2, pt3, zcoord, vals):
    """
    Adds a quadrilateral shape to sfwriter defined by the X,Y vertices
    pt0, pt1, pt2, and pt3, and possibly the common Z coordinate zcoord,
    along with the associated values in vals.

    Arguments:
       sfwriter - the shapefile.Writer object to add the shape and values to
       pt1, pt2,
       pt3, pt4 - the (X,Y) numeric coordinates of the vertices of the simple
                  quadrilateral; in sequence, but not necessarily the correct
                  winding.  Any coordinates after the first two in each point
                  are ignored.
       zcoord   - the numeric Z coordinate for this quadrilateral; may be None
       vals     - the collection of values to be associated with this shape
    """
    # Get the correct polygon type
    if zcoord != None:
       shapetype = shapefile.POLYGONZ
    else:
       shapetype = shapefile.POLYGON
    # Compute 2 * signed area of this simple quadrilateral
    dqarea  = pt0[0] * pt1[1] - pt1[0] * pt0[1]
    dqarea += pt1[0] * pt2[1] - pt2[0] * pt1[1]
    dqarea += pt2[0] * pt3[1] - pt3[0] * pt2[1]
    dqarea += pt3[0] * pt0[1] - pt0[0] * pt3[1]
    # Create the correctly ordered array of coordinates for this single shape part
    part = [ ]
    if dqarea < 0.0:
       # negative means clockwise which is desired direction
       part.append(list(pt0[:2]))
       part.append(list(pt1[:2]))
       part.append(list(pt2[:2]))
       part.append(list(pt3[:2]))
       part.append(list(pt0[:2]))
    else:
       # positive means counterclockwise so reverse ordering
       part.append(list(pt0[:2]))
       part.append(list(pt3[:2]))
       part.append(list(pt2[:2]))
       part.append(list(pt1[:2]))
       part.append(list(pt0[:2]))
    # Append the Z coordinate if given
    if zcoord != None:
       for pt in part:
          pt.append(zcoord)
    # Add the shape
    sfwriter.poly([ part, ], shapetype)
    # Add the values for this shape
    sfwriter.record(*vals)


#
# The following is only for testing this module from the command line
#
if __name__ == "__main__":
    import numpy
    import os

    shapefilename = "testsf"

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

    print "addquadxyvalues: SUCCESS"

