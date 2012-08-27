#! python
#

'''
Regridders designed for use in PyFerret, especially for Python external
functions for PyFerret.  Includes the singleton class ESMPControl to
safely start and stop ESMP once, and only once, in a Python session.

@author: Karl Smith
'''
import numpy

# Import classes given in modules in this package so they are all seen here.
try:
    from esmpcontrol import ESMPControl
    from regrid2d import CurvRectRegridder
except ImportError:
    # No ESMP, but do not raise an error until attempting to actually use it
    pass


def quadCornersFrom3D(ptx3d, pty3d):
    '''
    Converts 3D formats of curvilinear corner coordinates to 2D format.

    In 3D formats, quad[i,j] is formed by consecutive corner points
    in the sequence:
        (ptx3d[i,j,0], pty3d[i,j,0]),
        (ptx3d[i,j,1], pty3d[i,j,1]),
        (ptx3d[i,j,2], pty3d[i,j,2]),
        (ptx3d[i,j,3], pty3d[i,j,3]),
        (ptx3d[i,j,0], pty3d[i,j,0]),
    thus graphically represented, using the third indices, as:
          3 --- 2        1 --- 2
         /     /   or   /     /
        0 --- 1        0 --- 3
    Neighboring quadrilaterals must share edges, thus their 3D corner
    coordinates values must also satisfy:
        quad[i+1,j,0] is the same as quad[i,j,1] or quad[i,j,3]
        quad[i,j+j,0] is the same as quad[i,j,3] or quad[i,j,1]
        quad[i+1,j+1,0] is the same as quad[i,j,2]

    In the 2D format, quad[i,j] is former by consecutive corner points
    in the sequence:
        (ptx[i,   j],   pty[i,   j]) ,
        (ptx[i+1, j],   pty[i+1, j]) ,
        (ptx[i+1, j+1], pty[i+1, j+1]) ,
        (ptx[i,   j+1], pty[i,   j+1]) ,
        (ptx[i,   j],   pty[i,   j])
    The shape value of the first two dimensions for the 2D corner point
    arrays is one larger than the corresponding shape value for the 3D
    corner points.

    Arguments:
        ptx3d - 3D quadrilateral corner X coordinates
        pty3d - 3D quadrilateral corner Y coordinates
    Returns:
        (ptx, pty) where
        ptx - 2D quadrilateral corner X coordinates
        pty - 2D quadrilateral corner Y coordiantes
    Raises:
        ValueError if ptx3d or pty3d do not fit one of the expected formats
    '''
    # relative and absolute tolerances to be used in numpy.allclose
    rtol = 1.0E-4
    atol = 1.0E-6

    if ptx3d.shape != pty3d.shape:
        raise ValueError("ptx3d and pty3d in quadCornersFrom3D must have the same shape ")
    corners_shape = (ptx3d.shape[0] + 1, ptx3d.shape[1] + 1)

    if not (numpy.allclose(ptx3d[1:, 1:, 0], ptx3d[:-1, :-1, 2], rtol, atol) and \
             numpy.allclose(pty3d[1:, 1:, 0], pty3d[:-1, :-1, 2], rtol, atol)):
        raise ValueError("Unexpected ptx3d, pty3d values in quadCornersFrom3D")

    # Check if corners are: 
    #     3 --- 2        1 --- 2 
    #    /     /   or   /     / 
    #   0 --- 1        0 --- 3 
    # and assign appropriately. 
    if numpy.allclose(ptx3d[1:, :, 0], ptx3d[:-1, :, 1], rtol, atol) and \
       numpy.allclose(pty3d[1:, :, 0], pty3d[:-1, :, 1], rtol, atol) and \
       numpy.allclose(ptx3d[:, 1:, 0], ptx3d[:, :-1, 3], rtol, atol) and \
       numpy.allclose(pty3d[:, 1:, 0], pty3d[:, :-1, 3], rtol, atol):
        ptx = numpy.empty(corners_shape, dtype=numpy.float64)
        ptx[0, 0] = ptx3d[0, 0, 0]
        ptx[0, 1:] = ptx3d[0, :, 3]
        ptx[1:, 0] = ptx3d[:, 0, 1]
        ptx[1:, 1:] = ptx3d[:, :, 2]
        pty = numpy.empty(corners_shape, dtype=numpy.float64)
        pty[0, 0] = pty3d[0, 0, 0]
        pty[0, 1:] = pty3d[0, :, 3]
        pty[1:, 0] = pty3d[:, 0, 1]
        pty[1:, 1:] = pty3d[:, :, 2]
    elif numpy.allclose(ptx3d[1:, :, 0], ptx3d[:-1, :, 3], rtol, atol) and \
         numpy.allclose(pty3d[1:, :, 0], pty3d[:-1, :, 3], rtol, atol) and \
         numpy.allclose(ptx3d[:, 1:, 0], ptx3d[:, :-1, 1], rtol, atol) and \
         numpy.allclose(pty3d[:, 1:, 0], pty3d[:, :-1, 1], rtol, atol):
        ptx = numpy.empty(corners_shape, dtype=numpy.float64)
        ptx[0, 0] = ptx3d[0, 0, 0]
        ptx[0, 1:] = ptx3d[0, :, 1]
        ptx[1:, 0] = ptx3d[:, 0, 3]
        ptx[1:, 1:] = ptx3d[:, :, 2]
        pty = numpy.empty(corners_shape, dtype=numpy.float64)
        pty[0, 0] = pty3d[0, 0, 0]
        pty[0, 1:] = pty3d[0, :, 1]
        pty[1:, 0] = pty3d[:, 0, 3]
        pty[1:, 1:] = pty3d[:, :, 2]
    else:
        raise ValueError("Unexpected ptx3d, pty3d values in quadCornersFrom3D")

    return (ptx, pty)


def quadCentroids(ptx, pty):
    '''
    Returns the centroids of quadrilaterals using the Surveyor's formula.
    The quadilaterals are defined from the 2D corner point numpy arrays
    ptx and pty, which must have the same shape, where quad[i,j] is formed
    by joined consecutive x,y points in the sequence:
    ( ptx[i,   j],   pty[i,   j] ) ,
    ( ptx[i+1, j],   pty[i+1, j] ) ,
    ( ptx[i+1, j+1], pty[i+1, j+1] ) ,
    ( ptx[i,   j+1], pty[i,   j+1] ) ,
    ( ptx[i,   j],   pty[i,   j] )
    The ctrx and ctry arrays returned will have a shape one smaller in the
    first two dimensions compared to ptx and pty.
    '''
    if ptx.shape != pty.shape:
        raise ValueError("ptx and pty in quadCentroids must have the same shape")

    side0010 = ptx[:-1, :-1] * pty[1:, :-1] - ptx[1:, :-1] * pty[:-1, :-1]
    side1011 = ptx[1:, :-1] * pty[1:, 1:] - ptx[1:, 1:] * pty[1:, :-1]
    side1101 = ptx[1:, 1:] * pty[:-1, 1:] - ptx[:-1, 1:] * pty[1:, 1:]
    side0100 = ptx[:-1, 1:] * pty[:-1, :-1] - ptx[:-1, :-1] * pty[:-1, 1:]

    area = 0.5 * (side0010 + side1011 + side1101 + side0100)

    ctrx = (ptx[:-1, :-1] + ptx[1:, :-1]) * side0010
    ctrx += (ptx[1:, :-1] + ptx[1:, 1:]) * side1011
    ctrx += (ptx[1:, 1:] + ptx[:-1, 1:]) * side1101
    ctrx += (ptx[:-1, 1:] + ptx[:-1, :-1]) * side0100
    ctrx /= 6.0 * area

    ctry = (pty[:-1, :-1] + pty[1:, :-1]) * side0010
    ctry += (pty[1:, :-1] + pty[1:, 1:]) * side1011
    ctry += (pty[1:, 1:] + pty[:-1, 1:]) * side1101
    ctry += (pty[:-1, 1:] + pty[:-1, :-1]) * side0100
    ctry /= 6.0 * area

    return (ctrx, ctry)

