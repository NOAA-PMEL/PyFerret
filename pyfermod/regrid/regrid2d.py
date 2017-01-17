'''
Regridder for converting data between a curvilinear longitude,
latitude grid and a rectilinear longitude, latitude grid.
Uses the ESMP interface to ESMF to perform the regridding.

@author: Karl Smith
'''

import numpy
import ESMP


class CurvRectRegridder(object):
    '''
    Regridder for regridding data between a 2D curvilinear grid, where
    the longitude and latitude of each grid corner and/or center point
    is explicitly defined, and a 2D rectilinear grid, where the grid
    corners are all intersections of a given set of strictly increasing
    longitudes with a set of strictly increasing latitudes.

    For these grids, the center point [i, j] is taken to be the center
    point of the quadrilateral defined by connecting consecutive corner
    points in the sequence (corner_pt[i, j], corner_pt[i+1, j],
    corner_pt[i+1, j+1], corner_pt([i, j+1], corner_pt[i, j]).

    Uses the ESMP interface to ESMF to perform the regridding.  Prior
    to calling any instance methods in the CurvRectRegridder class, the
    ESMP module must be imported and ESMP.ESMP_Initialize() must have
    been called.  When a CurvRectRegridder instance is no longer needed,
    the finalize method of the instance should be called to free ESMP
    resources associated with the instance.  When ESMP is no longer
    required, the ESMP.ESMP_Finalize() method should be called to free
    all ESMP and ESMF resources.

    See the ESMPControl singleton class to simplify initializing and
    finalizing ESMP once, and only once, for a Python session.
    '''


    def __init__(self):
        '''
        Initializes to an empty regridder.  The ESMP module must be
        imported and ESMP.ESMP_Initialize() called (possibly through
        invoking ESMPControl().startCheckESMP()) prior to calling
        any methods in this instance.
        '''
        # tuples giving the shape of the grid (defined by number of cells)
        self.__curv_shape = None
        self.__rect_shape = None
        # ESMP_Grid objects describing the grids
        self.__curv_grid = None
        self.__rect_grid = None
        # ESMP_Field objects providing the source or regridded data
        self.__curv_dest_field = None
        self.__curv_src_field = None
        self.__rect_dest_field = None
        self.__rect_src_field = None
        # handles to regridding operations between the fields/grids
        self.__curv_to_rect_handles = { }
        self.__rect_to_curv_handles = { }


    def createCurvGrid(self, center_lons, center_lats, center_ignore=None,
                       corner_lons=None, corner_lats=None, corner_ignore=None):
        '''
        Create the curvilinear grid as an ESMP_Grid using the provided center
        longitudes and latitudes as the grid center points, and, if given,
        the grid corner longitudes and latitudes as the grid corner points.
        Curvilinear data is assigned to the center points.  Grid point
        coordinates are assigned as coord[i,j] = ( lons[i,j], lats[i,j] ).

        For these grids, the center point [i, j] is taken to be the center
        point of the quadrilateral defined by connecting consecutive corner
        points in the sequence (corner_pt[i, j], corner_pt[i+1, j],
        corner_pt[i+1, j+1], corner_pt([i ,j+1], corner_pt[i, j]).

        Any previous ESMP_Grid, ESMP_Field, or ESMP regridding procedures
        are destroyed.

        Arguments:
            center_lons:   2D array of longitudes, in degrees,
                           for each of the curvilinear center points
            center_lats:   2D array of latitudes, in degrees,
                           for each of the curvilinear center points
            center_ignore: 2D array of boolean-like values, indicating if
                           the corresponding grid center point should be
                           ignored in the regridding; if None, no grid
                           center points will be ignored
            corner_lons:   2D array of longitudes, in degrees,
                           for each of the curvilinear corner points
            corner_lats:   2D array of latitudes, in degrees,
                           for each of the curvilinear corner points
            corner_ignore: 2D array of boolean-like values, indicating if
                           the corresponding grid corner point should be
                           ignored in the regridding; if None, no grid
                           corner points will be ignored
        Returns:
            None
        Raises:
            ValueError: if the shape (dimensionality) of an argument is
                        invalid, or if a value in an argument is invalid
            TypeError:  if an argument is not array-like
        '''
        # Make sure center_lons is an appropriate array-like argument
        center_lons_array = numpy.array(center_lons, dtype=numpy.float64, copy=False)
        if len(center_lons_array.shape) != 2:
            raise ValueError("center_lons must be two-dimensional")

        # Make sure center_lats is an appropriate array-like argument
        center_lats_array = numpy.array(center_lats, dtype=numpy.float64, copy=False)
        if center_lats_array.shape != center_lons_array.shape:
            raise ValueError("center_lats and center_lons must have the same shape")

        if center_ignore == None:
            # Using all points, no mask created
            center_ignore_array = None
        else:
            # Make sure ignore_pts is an appropriate array-like argument
            center_ignore_array = numpy.array(center_ignore, dtype=numpy.bool, copy=False)
            if center_ignore_array.shape != center_lons_array.shape:
                raise ValueError("center_ignore and center_lons must have the same shape")
            # If not actually ignoring any points, do not create a mask
            if not center_ignore_array.any():
                center_ignore_array = None

        if (corner_lons != None) and (corner_lats != None):
            # Corner points specified

            # Make sure corner_lons is an appropriate array-like argument
            corner_lons_array = numpy.array(corner_lons, dtype=numpy.float64, copy=False)
            if corner_lons_array.shape != (center_lons_array.shape[0] + 1,
                                           center_lons_array.shape[1] + 1):
                raise ValueError("corner_lons must have one more point along " \
                                 "each dimension when compared to center_lons")

            # Make sure corner_lats is an appropriate array-like argument
            corner_lats_array = numpy.array(corner_lats, dtype=numpy.float64, copy=False)
            if corner_lats_array.shape != corner_lons_array.shape:
                raise ValueError("corner_lats and corner_lons must have the same shape")

            if corner_ignore == None:
                # Using all points, no mask created
                corner_ignore_array = None
            else:
                # Make sure ignore_pts is an appropriate array-like argument
                corner_ignore_array = numpy.array(corner_ignore, dtype=numpy.bool, copy=False)
                if corner_ignore_array.shape != corner_lons_array.shape:
                    raise ValueError("corner_ignore and corner_lons must have the same shape")
                # If not actually ignoring any points, do not create a mask
                if not corner_ignore_array.any():
                    corner_ignore_array = None

        elif corner_lons != None:
            raise ValueError("corner_lons given without corner_lats")
        elif corner_lats != None:
            raise ValueError("corner_lats given without corner_lons")
        elif corner_ignore != None:
            raise ValueError("corner_ignore given without corner_lons and corner_lats")
        else:
            # No corner points specified
            corner_lats_array = None
            corner_lons_array = None
            corner_ignore_array = None

        # If a centers mask is given and corner points are given, 
        # but a corners mask is not given, create a mask ignoring 
        # all corners around an ignored center
        if (corner_ignore_array == None) and (corner_lons_array != None) \
                and (center_ignore_array != None):
            corner_ignore_array = numpy.zeros((center_ignore_array.shape[0]+1, 
                                               center_ignore_array.shape[1]+1),
                                              dtype=numpy.bool, order='F')
            corner_ignore_array[:-1,:-1] = center_ignore_array
            corner_ignore_array[1:,:-1] = numpy.logical_or(corner_ignore_array[1:,:-1], 
                                                           center_ignore_array)
            corner_ignore_array[1:,1:] = numpy.logical_or(corner_ignore_array[1:,1:], 
                                                           center_ignore_array)
            corner_ignore_array[:-1,1:] = numpy.logical_or(corner_ignore_array[:-1,1:], 
                                                           center_ignore_array)

        # Release any regridding procedures and clear the dictionaries
        for handle in list(self.__rect_to_curv_handles.values()):
            ESMP.ESMP_FieldRegridRelease(handle)
        self.__rect_to_curv_handles.clear()
        for handle in list(self.__curv_to_rect_handles.values()):
            ESMP.ESMP_FieldRegridRelease(handle)
        self.__curv_to_rect_handles.clear()
        # Destroy any curvilinear ESMP_Fields
        if self.__curv_src_field != None:
            ESMP.ESMP_FieldDestroy(self.__curv_src_field)
            self.__curv_src_field = None
        if self.__curv_dest_field != None:
            ESMP.ESMP_FieldDestroy(self.__curv_dest_field)
            self.__curv_dest_field = None
        # Destroy any previous curvilinear ESMP_Grid
        if self.__curv_grid != None:
            ESMP.ESMP_GridDestroy(self.__curv_grid);
            self.__curv_grid = None

        # Create the curvilinear longitude, latitude ESMP_Grid using
        # ESMP_GridCreateNoPeriDim for the typical case (not the
        # whole world) in Ferret.
        self.__curv_shape = center_lons_array.shape
        grid_shape = numpy.array(self.__curv_shape, dtype=numpy.int32)
        self.__curv_grid = ESMP.ESMP_GridCreateNoPeriDim(grid_shape,
                                         ESMP.ESMP_COORDSYS_SPH_DEG,
                                         ESMP.ESMP_TYPEKIND_R8)

        if corner_lons_array != None:
            # Allocate space for the grid corner coordinates
            ESMP.ESMP_GridAddCoord(self.__curv_grid, ESMP.ESMP_STAGGERLOC_CORNER)

            # Retrieve the grid corner coordinate arrays in the ESMP_Grid
            grid_lon_coords = ESMP.ESMP_GridGetCoordPtr(self.__curv_grid, 0,
                                                        ESMP.ESMP_STAGGERLOC_CORNER)
            grid_lat_coords = ESMP.ESMP_GridGetCoordPtr(self.__curv_grid, 1,
                                                        ESMP.ESMP_STAGGERLOC_CORNER)

            # Assign the longitudes and latitudes of the grid corners in the ESMP_Grid
            grid_lon_coords[:] = corner_lons_array.flatten('F')
            grid_lat_coords[:] = corner_lats_array.flatten('F')

            # Add a mask if not considering all the corner points
            if corner_ignore_array != None:
                # Allocate space for the grid corners mask
                ESMP.ESMP_GridAddItem(self.__curv_grid, ESMP.ESMP_GRIDITEM_MASK,
                                                        ESMP.ESMP_STAGGERLOC_CORNER)
                # Retrieve the grid corners mask array in the ESMP_Grid
                ignore_mask = ESMP.ESMP_GridGetItem(self.__curv_grid,
                                                    ESMP.ESMP_GRIDITEM_MASK,
                                                    ESMP.ESMP_STAGGERLOC_CORNER)
                # Assign the mask in the ESMP_Grid; 
                # False (turns into zero) means use the point;
                # True (turns into one) means ignore the point
                ignore_mask[:] = corner_ignore_array.flatten('F')

        # Allocate space for the grid center coordinates
        ESMP.ESMP_GridAddCoord(self.__curv_grid, ESMP.ESMP_STAGGERLOC_CENTER)

        # Retrieve the grid center coordinate arrays in the ESMP_Grid
        grid_lon_coords = ESMP.ESMP_GridGetCoordPtr(self.__curv_grid, 0,
                                                    ESMP.ESMP_STAGGERLOC_CENTER)
        grid_lat_coords = ESMP.ESMP_GridGetCoordPtr(self.__curv_grid, 1,
                                                    ESMP.ESMP_STAGGERLOC_CENTER)

        # Assign the longitudes and latitudes of the grid centers in the ESMP_Grid
        grid_lon_coords[:] = center_lons_array.flatten('F')
        grid_lat_coords[:] = center_lats_array.flatten('F')

        # Add a mask if not considering all the center points
        if center_ignore_array != None:
            # Allocate space for the grid centers mask
            ESMP.ESMP_GridAddItem(self.__curv_grid, ESMP.ESMP_GRIDITEM_MASK,
                                                    ESMP.ESMP_STAGGERLOC_CENTER)
            # Retrieve the grid centers mask array in the ESMP_Grid
            ignore_mask = ESMP.ESMP_GridGetItem(self.__curv_grid,
                                                ESMP.ESMP_GRIDITEM_MASK,
                                                ESMP.ESMP_STAGGERLOC_CENTER)
            # Assign the mask in the ESMP_Grid; 
            # False (turns into zero) means use the point;
            # True (turns into one) means ignore the point
            ignore_mask[:] = center_ignore_array.flatten('F')


    def assignCurvField(self, data=None):
        '''
        Possibly creates, and possible assigns, an appropriate curvilinear
        ESMP_Field located at the center points of the grid.  An ESMP_Field
        is created if and only if it is not already present; thus allowing
        the "weights" computed from a previous regridding to be reused.

        If data is not None, assigns the data values to the curvilinear
        source ESMP_Field, creating the ESMP_field if it does not exist.

        If data is None, only creates the curvilinear destination ESMP_Field
        if it does not exist.

        Arguments:
            data: 2D array of data values to be assigned in an
                  ESMP_Field for the grid center points; if None,
                  no data in any ESMP_Field is modified.
        Returns:
            None
        Raises:
            ValueError: if data is not None, and if the shape
                        (dimensionality) of data is invalid or
                        if a value in data is not numeric
            TypeError:  if data, if not None, is not array-like
        '''
        if data == None:

            # Create the curvilinear destination ESMP_Field if it does not exist
            if self.__curv_dest_field == None:
                self.__curv_dest_field = ESMP.ESMP_FieldCreateGrid(self.__curv_grid,
                                                        "curv_dest_field",
                                                        ESMP.ESMP_TYPEKIND_R8,
                                                        ESMP.ESMP_STAGGERLOC_CENTER)

        else:

            # Make sure data is an appropriate array-like argument
            data_array = numpy.array(data, dtype=numpy.float64, copy=True)
            if data_array.shape != self.__curv_shape:
                raise ValueError("data must have the same shape " \
                                 "as the center points arrays")

            # Create the curvilinear source ESMP_Field if it does not exist
            if self.__curv_src_field == None:
                self.__curv_src_field = ESMP.ESMP_FieldCreateGrid(self.__curv_grid,
                                                       "curv_src_field",
                                                       ESMP.ESMP_TYPEKIND_R8,
                                                       ESMP.ESMP_STAGGERLOC_CENTER)

            # Retrieve the field data array in the ESMP_Field
            field_ptr = ESMP.ESMP_FieldGetPtr(self.__curv_src_field)

            # Assign the data to the field array in the ESMP_Field
            field_ptr[:] = data_array.flatten('F')


    def createRectGrid(self, center_lons, center_lats, center_ignore= None,
                       corner_lons=None, corner_lats=None, corner_ignore=None):
        '''
        Create the rectilinear grid as an ESMP_Grid using the provided center
        longitudes and latitudes as the grid center points, and, if given,
        the grid corner longitudes and latitudes as the grid corner points.
        Rectilinear data is assigned to the center points.  Grid point
        coordinates are assigned as coord[i,j] = ( lons[i], lats[j] ).

        For these grids, the center point [i,j] is taken to be the center
        point of the quadrilateral defined by connecting consecutive corner
        points in the sequence ( corner_pt[i, j], corner_pt[i+1, j],
        corner_pt[i+1, j+1], corner_pt([i, j+1], corner_pt[i, j] ).

        Any previous ESMP_Grid, ESMP_Field, or ESMP regridding procedures
        are destroyed.

        Arguments:
            center_lons:   1D array of longitudes, in degrees,
                           for each of the rectilinear grid cells
                           center planes
            center_lats:   1D array of latitudes, in degrees,
                           for each of the rectilinear grid cells
                           center planes
            center_ignore: 2D array of boolean-like values, indicating if
                           the corresponding grid center point should be
                           ignored in the regridding; if None, no grid
                           center points will be ignored
            corner_lons:   1D array of longitudes, in degrees,
                           for each of the rectilinear grid cells
                           corner planes
            corner_lats:   1D array of latitudes, in degrees,
                           for each of the rectilinear grid cells
                           corner planes
            corner_ignore: 2D array of boolean-like values, indicating if
                           the corresponding grid corner point should be
                           ignored in the regridding; if None, no grid
                           corner points will be ignored
        Returns:
            None
        Raises:
            ValueError: if the shape (dimensionality) of an argument is
                        invalid, or if a value in an argument is invalid
            TypeError:  if an argument is not array-like
        '''
        # Make sure center_lons is an appropriate array-like argument
        center_lons_array = numpy.array(center_lons, dtype=numpy.float64, copy=False)
        if len(center_lons_array.shape) != 1:
            raise ValueError("center_lons must be one-dimensional")

        # Make sure center_lats is an appropriate array-like argument
        center_lats_array = numpy.array(center_lats, dtype=numpy.float64, copy=True)
        if len(center_lats_array.shape) != 1:
            raise ValueError("center_lats must be one-dimensional")

        if center_ignore == None:
            # Using all center points, no mask created
            center_ignore_array = None
        else:
            # Make sure center_ignore is an appropriate array-like argument
            center_ignore_array = numpy.array(center_ignore, dtype=numpy.bool, copy=False)
            if center_ignore_array.shape != (center_lons_array.shape[0],
                                             center_lats_array.shape[0]):
                raise ValueError("center_ignore must have the shape " \
                                 "( len(center_lons), len(center_lats) )")
            # If not actually ignoring any center points, do not create a mask
            if not center_ignore_array.any():
                center_ignore_array = None

        if (corner_lons != None) and (corner_lats != None):
            # Corner points specified

            # Make sure corner_lons is an appropriate array-like argument
            corner_lons_array = numpy.array(corner_lons, dtype=numpy.float64, copy=False)
            if corner_lons_array.shape != (center_lons_array.shape[0] + 1, ):
                raise ValueError("corner_lons must have shape ( len(center_lons) + 1, )")

            # Make sure corner_lats is an appropriate array-like argument
            corner_lats_array = numpy.array(corner_lats, dtype=numpy.float64, copy=True)
            if corner_lats_array.shape != (center_lats_array.shape[0] + 1, ):
                raise ValueError("corner_lats must have shape ( len(center_lats) + 1, )")

            if corner_ignore == None:
                # Using all corner points, no mask created
                corner_ignore_array = None
            else:
                # Make sure corner_ignore is an appropriate array-like argument
                corner_ignore_array = numpy.array(corner_ignore, dtype=numpy.bool, copy=False)
                if corner_ignore_array.shape != (corner_lons_array.shape[0],
                                                 corner_lats_array.shape[0]):
                    raise ValueError("corner_ignore must have the shape " \
                                     "( len(corner_lons), len(corner_lats) )")
                # If not actually ignoring any points, do not create a mask
                if not corner_ignore_array.any():
                    corner_ignore_array = None

        elif corner_lons != None:
            raise ValueError("corner_lons given without corner_lats")
        elif corner_lats != None:
            raise ValueError("corner_lats given without corner_lons")
        elif corner_ignore != None:
            raise ValueError("corner_ignore given without corner_lons and corner_lats")
        else:
            # No corner points specified
            corner_lats_array = None
            corner_lons_array = None
            corner_ignore_array = None

        # If a centers mask is given and corner points are given, 
        # but a corners mask is not given, create a mask ignoring 
        # all corners around an ignored center
        if (corner_ignore_array == None) and (corner_lons_array != None) \
                and (center_ignore_array != None):
            corner_ignore_array = numpy.zeros((center_ignore_array.shape[0]+1, 
                                               center_ignore_array.shape[1]+1),
                                              dtype=numpy.bool, order='F')
            corner_ignore_array[:-1,:-1] = center_ignore_array
            corner_ignore_array[1:,:-1] = numpy.logical_or(corner_ignore_array[1:,:-1], 
                                                           center_ignore_array)
            corner_ignore_array[1:,1:] = numpy.logical_or(corner_ignore_array[1:,1:], 
                                                           center_ignore_array)
            corner_ignore_array[:-1,1:] = numpy.logical_or(corner_ignore_array[:-1,1:], 
                                                           center_ignore_array)

        # Release any regridding procedures and clear the dictionaries
        for handle in list(self.__rect_to_curv_handles.values()):
            ESMP.ESMP_FieldRegridRelease(handle)
        self.__rect_to_curv_handles.clear()
        for handle in list(self.__curv_to_rect_handles.values()):
            ESMP.ESMP_FieldRegridRelease(handle)
        self.__curv_to_rect_handles.clear()
        # Destroy any rectilinear ESMP_Fields
        if self.__rect_src_field != None:
            ESMP.ESMP_FieldDestroy(self.__rect_src_field)
            self.__rect_src_field = None
        if self.__rect_dest_field != None:
            ESMP.ESMP_FieldDestroy(self.__rect_dest_field)
            self.__rect_dest_field = None
        # Destroy any previous rectilinear ESMP_Grid
        if self.__rect_grid != None:
            ESMP.ESMP_GridDestroy(self.__rect_grid);
            self.__rect_grid = None

        # Create the rectilinear longitude, latitude grid using
        # ESMP_GridCreateNoPeriDim for the typical case (not the
        # whole world) in Ferret.
        self.__rect_shape = (center_lons_array.shape[0],
                             center_lats_array.shape[0])
        grid_shape = numpy.array(self.__rect_shape, dtype=numpy.int32)
        self.__rect_grid = ESMP.ESMP_GridCreateNoPeriDim(grid_shape,
                                         ESMP.ESMP_COORDSYS_SPH_DEG,
                                         ESMP.ESMP_TYPEKIND_R8)

        if corner_lons_array != None:
            # Allocate space for the grid corner coordinates
            ESMP.ESMP_GridAddCoord(self.__rect_grid, ESMP.ESMP_STAGGERLOC_CORNER)

            # Retrieve the grid corner coordinate arrays in the ESMP_Grid
            grid_lon_coords = ESMP.ESMP_GridGetCoordPtr(self.__rect_grid, 0,
                                                        ESMP.ESMP_STAGGERLOC_CORNER)
            grid_lat_coords = ESMP.ESMP_GridGetCoordPtr(self.__rect_grid, 1,
                                                        ESMP.ESMP_STAGGERLOC_CORNER)

            # Assign the longitudes and latitudes of the grid corners in the ESMP_Grid
            # numpy.tile([a,b,c], 2) = [a,b,c,a,b,c]
            grid_lon_coords[:] = numpy.tile(corner_lons_array, corner_lats_array.shape[0])
            # numpy.repeat([a,b,c], 2) = [a,a,b,b,c,c]
            grid_lat_coords[:] = numpy.repeat(corner_lats_array, corner_lons_array.shape[0])

            # Add a mask if not considering all the corner points
            if corner_ignore_array != None:
                # Allocate space for the grid corners mask
                ESMP.ESMP_GridAddItem(self.__rect_grid, ESMP.ESMP_GRIDITEM_MASK,
                                                        ESMP.ESMP_STAGGERLOC_CORNER)
                # Retrieve the grid corners mask array in the ESMP_Grid
                ignore_mask = ESMP.ESMP_GridGetItem(self.__rect_grid,
                                                    ESMP.ESMP_GRIDITEM_MASK,
                                                    ESMP.ESMP_STAGGERLOC_CORNER)
                # Assign the mask in the ESMP_Grid; 
                # False (turns into zero) means use the point;
                # True (turns into one) means ignore the point;
                # flatten in column-major (F) order to match lon-lat assignment order
                ignore_mask[:] = corner_ignore_array.flatten('F')

        # Allocate space for the grid center coordinates
        ESMP.ESMP_GridAddCoord(self.__rect_grid, ESMP.ESMP_STAGGERLOC_CENTER)

        # Retrieve the grid corner coordinate arrays in the ESMP_Grid
        grid_lon_coords = ESMP.ESMP_GridGetCoordPtr(self.__rect_grid, 0,
                                                    ESMP.ESMP_STAGGERLOC_CENTER)
        grid_lat_coords = ESMP.ESMP_GridGetCoordPtr(self.__rect_grid, 1,
                                                    ESMP.ESMP_STAGGERLOC_CENTER)

        # Assign the longitudes and latitudes of the grid centers in the ESMP_Grid
        # numpy.tile([a,b,c], 2) = [a,b,c,a,b,c]
        grid_lon_coords[:] = numpy.tile(center_lons_array, center_lats_array.shape[0])
        # numpy.repeat([a,b,c], 2) = [a,a,b,b,c,c]
        grid_lat_coords[:] = numpy.repeat(center_lats_array, center_lons_array.shape[0])

        # Add a mask if not considering all the center points
        if center_ignore_array != None:
            # Allocate space for the grid centers mask
            ESMP.ESMP_GridAddItem(self.__rect_grid, ESMP.ESMP_GRIDITEM_MASK,
                                                    ESMP.ESMP_STAGGERLOC_CENTER)
            # Retrieve the grid centers mask array in the ESMP_Grid
            ignore_mask = ESMP.ESMP_GridGetItem(self.__rect_grid,
                                                ESMP.ESMP_GRIDITEM_MASK,
                                                ESMP.ESMP_STAGGERLOC_CENTER)
            # Assign the mask in the ESMP_Grid; 
            # False (turns into zero) means use the point;
            # True (turns into one) means ignore the point;
            # flatten in column-major (F) order to match lon-lat assignment order
            ignore_mask[:] = center_ignore_array.flatten('F')


    def assignRectField(self, data=None):
        '''
        Possibly creates, and possible assigns, an appropriate rectilinear
        ESMP_Field located at the center points of the grid.  An ESMP_Field
        is created if and only if it is not already present; thus allowing
        the "weights" computed from a previous regridding to be reused.

        If data is not None, assigns the data values to the rectilinear
        source ESMP_Field, creating the ESMP_field if it does not exist.

        If data is None, only creates the rectilinear destination ESMP_Field
        if it does not exist.

        Arguments:
            data: 2D array of data values to be assigned in an
                  ESMP_Field for the grid center points; if None,
                  no data in any ESMP_Field is modified.
        Returns:
            None
        Raises:
            ValueError: if data is not None, and if the shape
                        (dimensionality) of data is invalid or
                        if a value in data is not numeric
            TypeError:  if data, if not None, is not array-like
        '''
        if data == None:

            # Create the rectilinear destination ESMP_Field if it does not exist
            if self.__rect_dest_field == None:
                self.__rect_dest_field = ESMP.ESMP_FieldCreateGrid(self.__rect_grid,
                                                        "rect_dest_field",
                                                        ESMP.ESMP_TYPEKIND_R8,
                                                        ESMP.ESMP_STAGGERLOC_CENTER)

        else:

            # Make sure data is an appropriate array-like argument
            data_array = numpy.array(data, dtype=numpy.float64, copy=True)
            if len(data_array.shape) != 2:
                raise ValueError("data must be two-dimensional")
            if data_array.shape != self.__rect_shape:
                raise ValueError("data must have the same shape " \
                                 "as the center points arrays")

            # Create the rectilinear source ESMP_Field if it does not exist
            if self.__rect_src_field == None:
                self.__rect_src_field = ESMP.ESMP_FieldCreateGrid(self.__rect_grid,
                                                       "rect_src_field",
                                                       ESMP.ESMP_TYPEKIND_R8,
                                                       ESMP.ESMP_STAGGERLOC_CENTER)

            # Retrieve the field data array in the ESMP_Field
            field_ptr = ESMP.ESMP_FieldGetPtr(self.__rect_src_field)

            # Assign the data to the field array in the ESMP_Field
            field_ptr[:] = data_array.flatten('F')


    def regridCurvToRect(self, undef_val,
                         method=ESMP.ESMP_REGRIDMETHOD_BILINEAR):
        '''
        Regrids from the curvilinear source ESMP_Field to the rectilinear
        destination ESMP_Field using the given regridding method.  Reuses
        the appropriate regridding procedure if one already exists;
        otherwise a new regridding procedure is created and stored.

        Prior to calling this method, the curvilinear source ESMP_Field
        must be created by calling createCurvGrid, then assignCurvField
        with valid data.  The rectilinear destination ESMP_Field must
        also have been created by calling createRectGrid, and then
        assignRectField with no data argument (or None for data).

        Arguments:
            undef_val: numpy array containing one numeric value to
                       be used as the undefined data value in the
                       returned array
            method:    one of the ESMP regridding method identifiers,
                       such as:
                           ESMP.ESMP_REGRIDMETHOD_BILINEAR
                           ESMP.ESMP_REGRIDMETHOD_CONSERVE
                           ESMP.ESMP_REGRIDMETHOD_PATCH
                       Conservative regridding requires that both
                       corner and center point coordinates are
                       defined in the grids.
        Returns:
            data: a 2D numpy array of data values located at the rectilinear
                  grid centers representing the regridded curvilinear
                  ESMP_Field data.  The undefined data value will be
                  assigned to unassigned data points.
        Raises:
            ValueError: if either the curvilinear source ESMP_Field or
                        the rectilinear destination ESMP_Field does not
                        exist.
        '''
        # Check that the source and destination fields exist
        if self.__curv_src_field == None:
            raise ValueError("Curvilinear source ESMP_Field does not exist")
        if self.__rect_dest_field == None:
            raise ValueError("Rectilinear destination ESMP_Field does not exist")
        # Check if a regrid procedure handle already exists for this method
        handle = self.__curv_to_rect_handles.get(method, None)
        # If no handle found, create one
        if handle == None:
            # Assign the value in the masks marking points to be ignored
            ignore_mask_value = numpy.array([1], dtype=numpy.int32)
            # Generate the procedure handle
            handle = ESMP.ESMP_FieldRegridStore(self.__curv_src_field,
                                                self.__rect_dest_field,
                                                ignore_mask_value, ignore_mask_value,
                                                method, ESMP.ESMP_UNMAPPEDACTION_IGNORE)
            # Save the handle for this method for future regrids
            self.__curv_to_rect_handles[method] = handle
        # Initialize the destination field values with the undefined data value
        field_ptr = ESMP.ESMP_FieldGetPtr(self.__rect_dest_field)
        field_ptr[:] = undef_val
        # Perform the regridding, zeroing out only the 
        # destination fields values that will be assigned
        ESMP.ESMP_FieldRegrid(self.__curv_src_field, self.__rect_dest_field,
                              handle, ESMP.ESMP_REGION_SELECT)
        # Make a copy of the destination field values to return, reshaped to 2D
        result = numpy.array(field_ptr, dtype=numpy.float64, copy=True)
        result = result.reshape(self.__rect_shape, order='F')

        return result


    def regridRectToCurv(self, undef_val,
                         method=ESMP.ESMP_REGRIDMETHOD_BILINEAR):
        '''
        Regrids from the rectilinear source ESMP_Field to the curvilinear
        destination ESMP_Field using the given regridding method.  Reuses
        the appropriate regridding procedure if one already exists;
        otherwise a new regridding procedure is created and stored.

        Prior to calling this method, the rectilinear source ESMP_Field
        must be created by calling createRectGrid, then assignRectField
        with valid data.  The curvilinear destination ESMP_Field must
        also have been created by calling createCurvGrid, and then
        assignCurvField with no data argument (or None for data).

        Arguments:
            undef_val: numpy array containing one numeric value to
                       be used as the undefined data value in the
                       returned array
            method:    one of the ESMP regridding method identifiers,
                       such as:
                           ESMP.ESMP_REGRIDMETHOD_BILINEAR
                           ESMP.ESMP_REGRIDMETHOD_CONSERVE
                           ESMP.ESMP_REGRIDMETHOD_PATCH
                       Conservative regridding requires that both
                       corner and center point coordinates are
                       defined in the grids.
        Returns:
            data: a 2D array of data values located at the curvilinear
                  grid centers representing the regridded rectilinear
                  ESMP_Field data.  The undefined data value will be
                  assigned to unassigned data points.
        Raises:
            ValueError: if either the rectilinear source ESMP_Field or
                        the curvilinear destination ESMP_Field does not
                        exist.
        '''
        # Check that the source and destination fields exist
        if self.__rect_src_field == None:
            raise ValueError("Rectilinear source ESMP_Field does not exist")
        if self.__curv_dest_field == None:
            raise ValueError("Curvilinear destination ESMP_Field does not exist")
        # Check if a regrid procedure handle already exists for this method
        handle = self.__rect_to_curv_handles.get(method, None)
        # If no handle found, create one
        if handle == None:
            # Assign the value in the masks marking points to be ignored
            ignore_mask_value = numpy.array([1], dtype=numpy.int32)
            # Generate the procedure handle
            handle = ESMP.ESMP_FieldRegridStore(self.__rect_src_field,
                                                self.__curv_dest_field,
                                                ignore_mask_value, ignore_mask_value,
                                                method, ESMP.ESMP_UNMAPPEDACTION_IGNORE)
            # Save the handle for this method for future regrids
            self.__rect_to_curv_handles[method] = handle
        # Initialize the destination field values with the undefined data value
        field_ptr = ESMP.ESMP_FieldGetPtr(self.__curv_dest_field)
        field_ptr[:] = undef_val
        # Perform the regridding, zeroing out only the 
        # destination fields values that will be assigned
        ESMP.ESMP_FieldRegrid(self.__rect_src_field, self.__curv_dest_field,
                              handle, ESMP.ESMP_REGION_SELECT)
        # Make a copy of the destination field values to return, reshaped to 2D
        result = numpy.array(field_ptr, dtype=numpy.float64, copy=True)
        result = result.reshape(self.__curv_shape, order='F')

        return result


    def finalize(self):
        '''
        Destroys any ESMP_Grid, ESMP_Field, and ESMP regridding
        procedures present in this instance.  If ESMP is no longer
        needed, ESMP.ESMP_Finalize() (possibly by invoking
        ESMPControl().stopESMP()) should be called to free any
        ESMP and ESMF resources.

        Arguments:
            None
        Returns:
            None
        '''
        # Release any regridding procedures and clear the dictionaries
        for handle in list(self.__rect_to_curv_handles.values()):
            ESMP.ESMP_FieldRegridRelease(handle)
        self.__rect_to_curv_handles.clear()
        for handle in list(self.__curv_to_rect_handles.values()):
            ESMP.ESMP_FieldRegridRelease(handle)
        self.__curv_to_rect_handles.clear()
        # Destroy any ESMP_Fields
        if self.__rect_src_field != None:
            ESMP.ESMP_FieldDestroy(self.__rect_src_field)
            self.__rect_src_field = None
        if self.__rect_dest_field != None:
            ESMP.ESMP_FieldDestroy(self.__rect_dest_field)
            self.__rect_dest_field = None
        if self.__curv_src_field != None:
            ESMP.ESMP_FieldDestroy(self.__curv_src_field)
            self.__curv_src_field = None
        if self.__curv_dest_field != None:
            ESMP.ESMP_FieldDestroy(self.__curv_dest_field)
            self.__curv_dest_field = None
        # Destroy any ESMP_Grids
        if self.__rect_grid != None:
            ESMP.ESMP_GridDestroy(self.__rect_grid);
            self.__rect_grid = None
        if self.__curv_grid != None:
            ESMP.ESMP_GridDestroy(self.__curv_grid);
            self.__curv_grid = None
        # clear the shape attributes just to be complete
        self.__rect_shape = None
        self.__curv_shape = None


