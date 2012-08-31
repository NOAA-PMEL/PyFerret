#! python
#

'''
Regridder for converting data between a curvilinear longitude,
latitude, depth (or elevation) grid and a rectilinear longitude,
latitude, depth (or elevation) grid.  Uses the ESMP interface
to ESMF to perform the regridding.

@author: Karl Smith
'''

import numpy
import ESMP

class CurvRect3DRegridder(object):
    '''
    Regridder for regridding data between a 3D curvilinear grid, where
    the longitude, latitude, and depth (or elevation) of each grid corner
    and/or center point is explicitly defined, and a 3D rectilinear grid,
    where the grid corners are all intersections of sets of strictly
    increasing longitudes, strictly increasing latitudes, and strictly
    increasing depths (or elevations).  The rectilinear grid centers are
    the intersections of averaged consecutive longitude pairs, averaged
    consecutive latitude pairs, and averages consecutive depth (or
    elevation) pairs.

    For these grids, the center point [i, j, k] is taken to be the center
    point of the quadrilaterally-faced hexahedron defined by the corner
    points: corner_pt[i, j, k], corner_pt[i+1, j, k], corner_pt[i+1, j+1, k],
    corner_pt([i, j+1, k], corner_pt[i, j, k+1], corner_pt[i+1, j, k+1],
    cornmer_pt[i+1, j+1, k+1], and corner_pt[i, j+1, k+1].

    Uses the ESMP interface to ESMF to perform the regridding.  Cartesian
    grids are used in ESMP to perform this regridding, assuming a spherical
    earth with radius 6371.007 kilometers.

    Prior to calling any instance methods in the CurvRect3DRegridder class,
    the ESMP module must be imported and ESMP.ESMP_Initialize() must have
    been called.  When a CurvRect3DRegridder instance is no longer needed,
    the finalize method of the instance should be called to free ESMP
    resources associated with the instance.  When ESMP is no longer
    required, the ESMP.ESMP_Finalize() method should be called to free
    all ESMP and ESMF resources.

    See the ESMPControl singleton class to simplify initializing ESMP
    once, and only once, and scheduling finalization of ESMP.  At this
    time, ESMP cannot be reinitialized after it is finalized.
    '''


    def __init__(self, earth_radius=6371.007):
        '''
        Initializes to an empty regridder.  The ESMP module must be
        imported and ESMP.ESMP_Initialize() called prior to calling
        any methods in this instance.

        Arguments:
            earth_rad_km: earth radius, in kilometers, to use in computing
                          cartesian coordinates for the regridding process.
                          The precision of this value is not critical and
                          the authalic radius given as the default should
                          suffice for most purposes.
        '''
        # Call the no-argument __init__ of the superclass
        super(CurvRect3DRegridder, self).__init__()
        # Radius of the earth, in meters, to use in regridding
        self.__earth_rad = earth_radius * 1000.0
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


    def createCurvGrid(self, center_lons, center_lats, center_levs,
                       levels_are_depths=True, center_ignore=None,
                       corner_lons=None, corner_lats=None,
                       corner_levs=None, corner_ignore=None):
        '''
        Create the curvilinear grid as an ESMP_Grid using the provided center
        longitudes, latitudes, and depths (or elevations) as the grid
        center points, and, if given, the grid corner longitudes, latitudes,
        and depths (or elevations) as the grid corner points.  Curvilinear
        data is assigned to the center points.  Grid point coordinate
        coord[i, j, k] is assigned from lon[i, j, k], lat[i, j, k], and
        levs[i, j, k]

        For these grids, the center point [i, j, k] is taken to be the center
        point of the quadrilaterally-faced hexahedron defined by the corner
        points: corner_pt[i, j, k], corner_pt[i+1, j, k], corner_pt[i+1, j+1, k],
        corner_pt([i, j+1, k], corner_pt[i, j, k+1], corner_pt[i+1, j, k+1],
        corner_pt[i+1, j+1, k+1], and corner_pt[i, j+1, k+1].

        Any previous ESMP_Grid, ESMP_Field, or ESMP regridding procedures
        are destroyed.

        Arguments:
            center_lons:       3D array of longitudes, in degrees,
                               for each of the curvilinear center points
            center_lats:       3D array of latitudes, in degrees,
                               for each of the curvilinear center points
            center_levs:       3D array of levels, in meters from the earth
                               radius, for each of the curvilinear center
                               points
            levels_are_depths: True - levels are depths and subtracted from
                                      the earth radius,
                               False - levels are elevations and added to
                                      the earth radius
            center_ignore:     3D array of boolean-like values, indicating
                               if the corresponding grid center point should
                               be ignored in the regridding; if None, no
                               grid center points will be ignored
            corner_lons:       3D array of longitudes, in degrees,
                               for each of the curvilinear corner points
            corner_lats:       3D array of latitudes, in degrees,
                               for each of the curvilinear corner points
            corner_levs:       3D array of levels, in meters from the earth radius,
                               for each of the curvilinear corner points
            corner_ignore:     3D array of boolean-like values, indicating
                               if the corresponding grid corner point should
                               be ignored in the regridding; if None, no
                               grid corner points will be ignored
        Returns:
            None
        Raises:
            ValueError: if the shape (dimensionality) of an argument
                        is invalid, or if a value in an argument is invalid
            TypeError:  if an argument is not array-like
        '''
        # Make sure center_lons is an appropriate array-like argument
        center_lons_array = numpy.array(center_lons, dtype=numpy.float64, copy=False)
        if len(center_lons_array.shape) != 3:
            raise ValueError("center_lons must be three-dimensional")

        # Make sure center_lats is an appropriate array-like argument
        center_lats_array = numpy.array(center_lats, dtype=numpy.float64, copy=False)
        if len(center_lats_array.shape) != 3:
            raise ValueError("center_lats must be three-dimensional")
        if center_lats_array.shape != center_lons_array.shape:
            raise ValueError("center_lats and center_lons must have the same shape")

        # Make sure center_lats is an appropriate array-like argument
        center_levs_array = numpy.array(center_levs, dtype=numpy.float64, copy=False)
        if len(center_levs_array.shape) != 3:
            raise ValueError("center_levs must be three-dimensional")
        if center_levs_array.shape != center_lons_array.shape:
            raise ValueError("center_levs and center_lons must have the same shape")

        if center_ignore == None:
            # Using all points, no mask created
            center_ignore_array = None
        else:
            # Make sure ignore_pts is an appropriate array-like argument
            center_ignore_array = numpy.array(center_ignore, dtype=numpy.bool, copy=False)
            if len(center_ignore_array.shape) != 3:
                raise ValueError("center_ignore must be three-dimensional")
            if center_ignore_array.shape != center_lons_array.shape:
                raise ValueError("center_ignore and center_lons must have the same shape")
            # If not actually ignoring any points, do not create a mask
            if not center_ignore_array.any():
                center_ignore_array = None

        if (corner_lons != None) and (corner_lats != None) and (corner_levs != None):
            # Corner points specified

            # Make sure corner_lons is an appropriate array-like argument
            corner_lons_array = numpy.array(corner_lons, dtype=numpy.float64, copy=False)
            if len(corner_lons_array.shape) != 3:
                raise ValueError("corner_lons must be three-dimensional")
            if corner_lons_array.shape != (center_lons_array.shape[0] + 1,
                                           center_lons_array.shape[1] + 1,
                                           center_lons_array.shape[2] + 1):
                raise ValueError("corner_lons must have one more point along " \
                                 "each dimension when compared to center_lons")

            # Make sure corner_lats is an appropriate array-like argument
            corner_lats_array = numpy.array(corner_lats, dtype=numpy.float64, copy=False)
            if len(corner_lats_array.shape) != 3:
                raise ValueError("corner_lats must be three-dimensional")
            if corner_lats_array.shape != corner_lons_array.shape:
                raise ValueError("corner_lats and corner_lons must have the same shape")

            # Make sure corner_lats is an appropriate array-like argument
            corner_levs_array = numpy.array(corner_levs, dtype=numpy.float64, copy=False)
            if len(corner_levs_array.shape) != 3:
                raise ValueError("corner_lats must be three-dimensional")
            if corner_levs_array.shape != corner_lons_array.shape:
                raise ValueError("corner_levs and corner_lons must have the same shape")

            if corner_ignore == None:
                # Using all points, no mask created
                corner_ignore_array = None
            else:
                # Make sure ignore_pts is an appropriate array-like argument
                corner_ignore_array = numpy.array(corner_ignore, dtype=numpy.bool, copy=False)
                if len(corner_ignore_array.shape) != 3:
                    raise ValueError("corner_ignore must be three-dimensional")
                if corner_ignore_array.shape != corner_lons_array.shape:
                    raise ValueError("corner_ignore and corner_lons must have the same shape")
                # If not actually ignoring any points, do not create a mask
                if not corner_ignore_array.any():
                    corner_ignore_array = None

        elif (corner_lons != None) or (corner_lats != None) or (corner_levs != None):
            raise ValueError("one or two, but not all three, of corner_lons, corner_lats, and corner_levs are given")
        elif corner_ignore != None:
            raise ValueError("corner_ignore given without corner_lons, corner_lats, and corner_levs")
        else:
            # No corner points specified
            corner_lats_array = None
            corner_lons_array = None
            corner_levs_array = None
            corner_ignore_array = None

        # Release any regridding procedures and clear the dictionaries
        for handle in self.__rect_to_curv_handles.values():
            ESMP.ESMP_FieldRegridRelease(handle)
        self.__rect_to_curv_handles.clear()
        for handle in self.__curv_to_rect_handles.values():
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

        # Create the curvilinear 3D cartesian coordinates ESMP_Grid
        # using ESMP_GridCreateNoPeriDim for the typical case in Ferret.
        # ESMP_GridCreate1PeriDim assumes that the full globe is to be 
        # used; that there is a center point provided for a cell between
        # the last longitude and the first longitude and thus interpolates 
        # through the last and first longitude. 
        self.__curv_shape = center_lons_array.shape
        grid_shape = numpy.array(self.__curv_shape, dtype=numpy.int32)
        self.__curv_grid = ESMP.ESMP_GridCreateNoPeriDim(grid_shape,
                                         ESMP.ESMP_COORDSYS_CART,
                                         ESMP.ESMP_TYPEKIND_R8)

        if corner_lons_array != None:
            # Allocate space for the grid corner coordinates
            ESMP.ESMP_GridAddCoord(self.__curv_grid, ESMP.ESMP_STAGGERLOC_CORNER_VFACE)

            # Retrieve the grid corner coordinate arrays in the ESMP_Grid
            grid_x_coords = ESMP.ESMP_GridGetCoordPtr(self.__curv_grid, 0,
                                                      ESMP.ESMP_STAGGERLOC_CORNER_VFACE)
            grid_y_coords = ESMP.ESMP_GridGetCoordPtr(self.__curv_grid, 1,
                                                      ESMP.ESMP_STAGGERLOC_CORNER_VFACE)
            grid_z_coords = ESMP.ESMP_GridGetCoordPtr(self.__curv_grid, 2,
                                                      ESMP.ESMP_STAGGERLOC_CORNER_VFACE)

            # Assign the cartesian coordinates of the grid corners in the ESMP_Grid
            theta = numpy.deg2rad(corner_lons_array.flatten('F'))
            phi = numpy.deg2rad(corner_lats_array.flatten('F'))
            rad = corner_levs_array.flatten('F').copy()
            if levels_are_depths:
                rad *= -1.0
            rad += self.__earth_rad
            # XY plane through the equator, Z toward north pole
            # 0 lon, 0 lat = (R, 0, 0); 90 lon, 0 lat = (0, R, 0); * lon, 90 lat = (0, 0, R)
            grid_x_coords[:] = rad * numpy.cos(theta) * numpy.cos(phi)
            grid_y_coords[:] = rad * numpy.sin(theta) * numpy.cos(phi)
            grid_z_coords[:] = rad * numpy.sin(phi)

            # Add a mask if not considering all the corner points
            if corner_ignore_array != None:
                # Allocate space for the grid corners mask
                ESMP.ESMP_GridAddItem(self.__curv_grid, ESMP.ESMP_GRIDITEM_MASK,
                                                        ESMP.ESMP_STAGGERLOC_CORNER_VFACE)
                # Retrieve the grid corners mask array in the ESMP_Grid
                ignore_mask = ESMP.ESMP_GridGetItem(self.__curv_grid,
                                                    ESMP.ESMP_GRIDITEM_MASK,
                                                    ESMP.ESMP_STAGGERLOC_CORNER_VFACE)
                # Assign the mask in the ESMP_Grid; 
                # False (turns into zero) means use the point;
                # True (turns into one) means ignore the point
                ignore_mask[:] = corner_ignore_array.flatten('F')

        # Allocate space for the grid center coordinates
        ESMP.ESMP_GridAddCoord(self.__curv_grid, ESMP.ESMP_STAGGERLOC_CENTER_VCENTER)

        # Retrieve the grid center coordinate arrays in the ESMP_Grid
        grid_x_coords = ESMP.ESMP_GridGetCoordPtr(self.__curv_grid, 0,
                                                  ESMP.ESMP_STAGGERLOC_CENTER_VCENTER)
        grid_y_coords = ESMP.ESMP_GridGetCoordPtr(self.__curv_grid, 1,
                                                  ESMP.ESMP_STAGGERLOC_CENTER_VCENTER)
        grid_z_coords = ESMP.ESMP_GridGetCoordPtr(self.__curv_grid, 2,
                                                  ESMP.ESMP_STAGGERLOC_CENTER_VCENTER)

        # Assign the cartesian coordinates of the grid corners in the ESMP_Grid
        theta = numpy.deg2rad(center_lons_array.flatten('F'))
        phi = numpy.deg2rad(center_lats_array.flatten('F'))
        rad = center_levs_array.flatten('F').copy()
        if levels_are_depths:
            rad *= -1.0
        rad += self.__earth_rad
        # XY plane through the equator, Z toward north pole
        # 0 lon, 0 lat = (R, 0, 0); 90 lon, 0 lat = (0, R, 0); * lon, 90 lat = (0, 0, R)
        grid_x_coords[:] = rad * numpy.cos(theta) * numpy.cos(phi)
        grid_y_coords[:] = rad * numpy.sin(theta) * numpy.cos(phi)
        grid_z_coords[:] = rad * numpy.sin(phi)

        # Add a mask if not considering all the center points
        if center_ignore_array != None:
            # Allocate space for the grid centers mask
            ESMP.ESMP_GridAddItem(self.__curv_grid, ESMP.ESMP_GRIDITEM_MASK,
                                                    ESMP.ESMP_STAGGERLOC_CENTER_VCENTER)
            # Retrieve the grid centers mask array in the ESMP_Grid
            ignore_mask = ESMP.ESMP_GridGetItem(self.__curv_grid,
                                                ESMP.ESMP_GRIDITEM_MASK,
                                                ESMP.ESMP_STAGGERLOC_CENTER_VCENTER)
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
            data: 3D array of data values to be assigned in an
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
                                                        ESMP.ESMP_STAGGERLOC_CENTER_VCENTER)

        else:

            # Make sure data is an appropriate array-like argument
            data_array = numpy.array(data, dtype=numpy.float64, copy=True)
            if len(data_array.shape) != 3:
                raise ValueError("data must be three-dimensional")
            if data_array.shape != self.__curv_shape:
                raise ValueError("data must have the same shape " \
                                 "as the center points arrays")

            # Create the curvilinear source ESMP_Field if it does not exist
            if self.__curv_src_field == None:
                self.__curv_src_field = ESMP.ESMP_FieldCreateGrid(self.__curv_grid,
                                                       "curv_src_field",
                                                       ESMP.ESMP_TYPEKIND_R8,
                                                       ESMP.ESMP_STAGGERLOC_CENTER_VCENTER)

            # Retrieve the field data array in the ESMP_Field
            field_ptr = ESMP.ESMP_FieldGetPtr(self.__curv_src_field)

            # Assign the data to the field array in the ESMP_Field
            field_ptr[:] = data_array.flatten('F')


    def createRectGrid(self, edge_lons, edge_lats, edge_levs, levels_are_depths=True,
                       center_ignore=None, corner_ignore=None):
        '''
        Create the rectilinear grid as an ESMP_Grid using the provided cell
        edge longitudes, latitudes, and levels to define the grid center and
        corner points.  Rectilinear data is assigned to the center points.
        Grid corner point [i, j, k] is assigned from edge_lons[i], edge_lats[j],
        and edge_levs[k].  Grid center point [i, j, k] is assigned from
        (edge_lons[i] + edge_lons[i+1]) / 2, (edge_lats[j] + edge_lats[j+1]) / 2,
        and (edge_levs[k] + edge_levs[k+1]) / 2.

        Any previous ESMP_Grid, ESMP_Field, or ESMP regridding procedures
        are destroyed.

        Arguments:
            edge_lons:         1D array of longitudes, in degrees,
                               for each of the rectilinear grid cells edges;
                               must be strictly increasing values
            edge_lats:         1D array of latitudes, in degrees,
                               for each of the rectilinear grid cells edges;
                               must be strictly increasing values
            edge_levs:         1D array of levels, in meters from the earth
                               radius, for each of the rectilinear grid cells
                               edges; must be strictly increasing values
            levels_are_depths: True - levels are depths and subtracted from
                                      the earth radius,
                               False - levels are elevations and added to
                                      the earth radius
            center_ignore:     3D array of boolean-like values, indicating if
                               the corresponding grid center point should be
                               ignored in the regridding; if None, no grid
                               center points will be ignored
            corner_ignore:     3D array of boolean-like values, indicating if
                               the corresponding grid corner point should be
                               ignored in the regridding; if None, no grid
                               corner points will be ignored
        Returns:
            None
        Raises:
            ValueError: if the shape (dimensionality) of an argument
                        is invalid, or if a value in an argument is invalid
            TypeError:  if an argument is not array-like
        '''
        # Make sure edge_lons is an appropriate array-like argument
        lons_array = numpy.array(edge_lons, dtype=numpy.float64, copy=False)
        if len(lons_array.shape) != 1:
            raise ValueError("edge_lons must be one-dimensional")
        increasing = (lons_array[:-1] < lons_array[1:])
        if not increasing.all():
            raise ValueError("edge_lons must contain strictly increasing values")

        # Make sure edge_lats is an appropriate array-like argument
        lats_array = numpy.array(edge_lats, dtype=numpy.float64, copy=True)
        if len(lats_array.shape) != 1:
            raise ValueError("edge_lats must be one-dimensional")
        increasing = (lats_array[:-1] < lats_array[1:])
        if not increasing.all():
            raise ValueError("edge_lats must contain strictly increasing values")

        # Make sure edge_levs is an appropriate array-like argument
        levs_array = numpy.array(edge_levs, dtype=numpy.float64, copy=True)
        if len(levs_array.shape) != 1:
            raise ValueError("edge_levs must be one-dimensional")
        increasing = (levs_array[:-1] < lats_array[1:])
        if not increasing.all():
            raise ValueError("edge_levs must contain strictly increasing values")

        if center_ignore == None:
            # Using all center points, no mask created
            center_ignore_array = None
        else:
            # Make sure center_ignore is an appropriate array-like argument
            center_ignore_array = numpy.array(center_ignore, dtype=numpy.bool, copy=False)
            if len(center_ignore_array.shape) != 3:
                raise ValueError("center_ignore must be three-dimensional")
            if (center_ignore_array.shape[0] != lons_array.shape[0] - 1) or \
               (center_ignore_array.shape[1] != lats_array.shape[0] - 1) or \
               (center_ignore_array.shape[2] != levs_array.shape[0] - 1):
                raise ValueError("center_ignore must have the shape " \
                                 "( len(edge_lons) - 1, len(edge_lats) - 1, len(edge_levs) - 1 )")
            # If not actually ignoring any center points, do not create a mask
            if not center_ignore_array.any():
                center_ignore_array = None

        if corner_ignore == None:
            # Using all corner points, no mask created
            corner_ignore_array = None
        else:
            # Make sure corner_ignore is an appropriate array-like argument
            corner_ignore_array = numpy.array(corner_ignore, dtype=numpy.bool, copy=False)
            if len(corner_ignore_array.shape) != 3:
                raise ValueError("corner_ignore must be three-dimensional")
            if (corner_ignore_array.shape[0] != lons_array.shape[0]) or \
               (corner_ignore_array.shape[1] != lats_array.shape[0]) or \
               (corner_ignore_array.shape[2] != levs_array.shape[0]):
                raise ValueError("corner_ignore must have the shape " \
                                 "( len(edge_lons), len(edge_lats), len(edge_levs) )")
            # If not actually ignoring any points, do not create a mask
            if not corner_ignore_array.any():
                corner_ignore_array = None

        # Release any regridding procedures and clear the dictionaries
        for handle in self.__rect_to_curv_handles.values():
            ESMP.ESMP_FieldRegridRelease(handle)
        self.__rect_to_curv_handles.clear()
        for handle in self.__curv_to_rect_handles.values():
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

        # Create the rectilinear 3D cartesian coordinates ESMP_Grid
        # using ESMP_GridCreateNoPeriDim for the typical case in Ferret.
        # ESMP_GridCreate1PeriDim assumes that the full globe is to be 
        # used; that there is a center point provided for a cell between
        # the last longitude and the first longitude and thus interpolates 
        # through the last and first longitude. 
        self.__rect_shape = (lons_array.shape[0] - 1,
                             lats_array.shape[0] - 1,
                             levs_array.shape[0] - 1)
        grid_shape = numpy.array(self.__rect_shape, dtype=numpy.int32)
        self.__rect_grid = ESMP.ESMP_GridCreateNoPeriDim(grid_shape,
                                         ESMP.ESMP_COORDSYS_CART,
                                         ESMP.ESMP_TYPEKIND_R8)

        # Allocate space for the grid corner coordinates
        ESMP.ESMP_GridAddCoord(self.__rect_grid, ESMP.ESMP_STAGGERLOC_CORNER_VFACE)

        # Retrieve the grid corner coordinate arrays in the ESMP_Grid
        grid_x_coords = ESMP.ESMP_GridGetCoordPtr(self.__rect_grid, 0,
                                                  ESMP.ESMP_STAGGERLOC_CORNER_VFACE)
        grid_y_coords = ESMP.ESMP_GridGetCoordPtr(self.__rect_grid, 1,
                                                  ESMP.ESMP_STAGGERLOC_CORNER_VFACE)
        grid_z_coords = ESMP.ESMP_GridGetCoordPtr(self.__rect_grid, 2,
                                                  ESMP.ESMP_STAGGERLOC_CORNER_VFACE)

        # Assign the cartesian coordinates of the grid corners in the ESMP_Grid
        # numpy.tile([a,b,c], 2) = [a,b,c,a,b,c]
        # numpy.repeat([a,b,c], 2) = [a,a,b,b,c,c]
        theta = numpy.deg2rad(lons_array)
        cos_theta = numpy.tile(numpy.cos(theta), lats_array.shape[0] * levs_array.shape[0])
        sin_theta = numpy.tile(numpy.sin(theta), lats_array.shape[0] * levs_array.shape[0])
        phi = numpy.deg2rad(lats_array)
        cos_phi = numpy.tile(numpy.repeat(numpy.cos(phi), lons_array.shape[0]), levs_array.shape[0])
        sin_phi = numpy.tile(numpy.repeat(numpy.sin(phi), lons_array.shape[0]), levs_array.shape[0])
        rad = levs_array.copy()
        if levels_are_depths:
            rad *= -1.0
        rad += self.__earth_rad
        rad = numpy.repeat(rad, lons_array.shape[0] * lats_array.shape[0])
        # XY plane through the equator, Z toward north pole
        # 0 lon, 0 lat = (R, 0, 0); 90 lon, 0 lat = (0, R, 0); * lon, 90 lat = (0, 0, R)
        grid_x_coords[:] = rad * cos_theta * cos_phi
        grid_y_coords[:] = rad * sin_theta * cos_phi
        grid_z_coords[:] = rad * sin_phi

        # Add a mask if not considering all the corner points
        if (corner_ignore_array != None):
            # Allocate space for the grid corners mask
            ESMP.ESMP_GridAddItem(self.__rect_grid, ESMP.ESMP_GRIDITEM_MASK,
                                                    ESMP.ESMP_STAGGERLOC_CORNER_VFACE)
            # Retrieve the grid corners mask array in the ESMP_Grid
            ignore_mask = ESMP.ESMP_GridGetItem(self.__rect_grid,
                                                ESMP.ESMP_GRIDITEM_MASK,
                                                ESMP.ESMP_STAGGERLOC_CORNER_VFACE)
            # Assign the mask in the ESMP_Grid; 
            # False (turns into zero) means use the point;
            # True (turns into one) means ignore the point;
            # flatten in column-major (F) order to match lon-lat-lev assignment order
            ignore_mask[:] = corner_ignore_array.flatten('F')

        # Allocate space for the grid center coordinates
        ESMP.ESMP_GridAddCoord(self.__rect_grid, ESMP.ESMP_STAGGERLOC_CENTER_VCENTER)

        # Retrieve the grid corner coordinate arrays in the ESMP_Grid
        grid_x_coords = ESMP.ESMP_GridGetCoordPtr(self.__rect_grid, 0,
                                                  ESMP.ESMP_STAGGERLOC_CENTER_VCENTER)
        grid_y_coords = ESMP.ESMP_GridGetCoordPtr(self.__rect_grid, 1,
                                                  ESMP.ESMP_STAGGERLOC_CENTER_VCENTER)
        grid_z_coords = ESMP.ESMP_GridGetCoordPtr(self.__rect_grid, 2,
                                                  ESMP.ESMP_STAGGERLOC_CENTER_VCENTER)

        # Assign the cartesian coordinates of the grid centers in the ESMP_Grid
        # numpy.tile([a,b,c], 2) = [a,b,c,a,b,c]
        # numpy.repeat([a,b,c], 2) = [a,a,b,b,c,c]
        mid_lons = 0.5 * (lons_array[:-1] + lons_array[1:])
        mid_lats = 0.5 * (lats_array[:-1] + lats_array[1:])
        mid_levs = 0.5 * (levs_array[:-1] + levs_array[1:])
        theta = numpy.deg2rad(mid_lons)
        cos_theta = numpy.tile(numpy.cos(theta), mid_lats.shape[0] * mid_levs.shape[0])
        sin_theta = numpy.tile(numpy.sin(theta), mid_lats.shape[0] * mid_levs.shape[0])
        phi = numpy.deg2rad(mid_lats)
        cos_phi = numpy.tile(numpy.repeat(numpy.cos(phi), mid_lons.shape[0]), mid_levs.shape[0])
        sin_phi = numpy.tile(numpy.repeat(numpy.sin(phi), mid_lons.shape[0]), mid_levs.shape[0])
        rad = mid_levs.copy()
        if levels_are_depths:
            rad *= -1.0
        rad += self.__earth_rad
        rad = numpy.repeat(rad, mid_lons.shape[0] * mid_lats.shape[0])
        # XY plane through the equator, Z toward north pole
        # 0 lon, 0 lat = (R, 0, 0); 90 lon, 0 lat = (0, R, 0); * lon, 90 lat = (0, 0, R)
        grid_x_coords[:] = rad * cos_theta * cos_phi
        grid_y_coords[:] = rad * sin_theta * cos_phi
        grid_z_coords[:] = rad * sin_phi

        # Add a mask if not considering all the center points
        if (center_ignore_array != None):
            # Allocate space for the grid centers mask
            ESMP.ESMP_GridAddItem(self.__rect_grid, ESMP.ESMP_GRIDITEM_MASK,
                                                    ESMP.ESMP_STAGGERLOC_CENTER_VCENTER)
            # Retrieve the grid centers mask array in the ESMP_Grid
            ignore_mask = ESMP.ESMP_GridGetItem(self.__rect_grid,
                                                ESMP.ESMP_GRIDITEM_MASK,
                                                ESMP.ESMP_STAGGERLOC_CENTER_VCENTER)
            # Assign the mask in the ESMP_Grid; 
            # False (turns into zero) means use the point;
            # True (turns into one) means ignore the point;
            # flatten in column-major (F) order to match lon-lat-lev assignment order
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
            data: 3D array of data values to be assigned in an
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
                                                        ESMP.ESMP_STAGGERLOC_CENTER_VCENTER)

        else:

            # Make sure data is an appropriate array-like argument
            data_array = numpy.array(data, dtype=numpy.float64, copy=True)
            if len(data_array.shape) != 3:
                raise ValueError("data must be three-dimensional")
            if data_array.shape != self.__rect_shape:
                raise ValueError("data must have the same shape " \
                                 "as the center points arrays")

            # Create the rectilinear source ESMP_Field if it does not exist
            if self.__rect_src_field == None:
                self.__rect_src_field = ESMP.ESMP_FieldCreateGrid(self.__rect_grid,
                                                       "rect_src_field",
                                                       ESMP.ESMP_TYPEKIND_R8,
                                                       ESMP.ESMP_STAGGERLOC_CENTER_VCENTER)

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
            data: a 3D numpy array of data values located at the rectilinear
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
        # Make a copy of the destination field values to return, reshaped to 3D
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
            data: a 3D array of data values located at the curvilinear
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
        # Make a copy of the destination field values to return, reshaped to 3D
        result = numpy.array(field_ptr, dtype=numpy.float64, copy=True)
        result = result.reshape(self.__curv_shape, order='F')

        return result


    def finalize(self):
        '''
        Destroys any ESMP_Grid, ESMP_Field, and ESMP regridding
        procedures present in this instance.  If ESMP is no longer
        needed, ESMP.ESMP_Finalize() should be called to free any
        ESMP and ESMF resources.

        Arguments:
            None
        Returns:
            None
        '''
        # Release any regridding procedures and clear the dictionaries
        for handle in self.__rect_to_curv_handles.values():
            ESMP.ESMP_FieldRegridRelease(handle)
        self.__rect_to_curv_handles.clear()
        for handle in self.__curv_to_rect_handles.values():
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


###
### The following is for testing "by-hand" and to serve as a usage example
###

# Gulf of Mexico Relief of the Surface of the Earth (meters) 98W:82W:0.5 (columns), 18N:31N:0.5 (rows)
__GULF_MEX_ROSE = (
      ( 1627.1,  2130.9,  1317.9,   373.8,   125.7,    45.0,    25.0,    20.0,    25.0,    25.0,
          15.0,    15.0,    25.0,    25.0,    40.0,    74.3,   169.7,   179.7,   151.3,    23.0,
          -3.2, -1638.9, -3668.0, -4098.9, -3075.3, -3017.9, -4554.6, -4969.7, -4961.6, -5008.2,
       -4759.7, -4932.3, -4016.7, ),
      ( 1951.7,  1669.0,   685.7,    63.4,    50.7,   779.4,   400.0,   -70.7,   -49.0,   -16.2,
          -6.1,     2.0,    -9.6,     5.0,    69.7,    80.1,   194.7,   210.2,   110.6,     6.1,
          -0.3,  -192.5, -1587.3, -4357.8, -4401.9, -4397.7, -2980.1, -2035.6,  -463.5, -2265.7,
       -4933.4, -4959.4, -4490.4, ),
      ( 2220.5,  2654.2,  1232.2,    60.8,   -10.0,  -217.9, -1170.2,  -807.9,  -599.1,  -388.8,
        -101.1,   -40.1,   -14.0,    -8.1,    19.9,    65.0,   139.4,   175.8,   115.2,    40.3,
           0.0,  -220.1,  -919.6, -4284.3, -4502.4, -4474.9, -4418.9, -3648.0, -2258.9, -2581.0,
       -2998.4, -3578.1, -3493.9, ),
      ( 2593.1,  2368.9,  1097.8,    29.9,  -225.9, -1889.3, -2357.3, -1504.2, -1161.4,  -939.7,
        -702.9,  -141.0,   -47.9,   -29.0,   -13.1,    29.7,    94.8,   150.0,   110.0,    70.5,
           0.0,  -340.6, -1551.1, -4353.5, -4516.0, -4534.5, -4455.3, -4438.0, -4153.9, -3078.7,
       -3004.0, -3228.8, -3595.4, ),
      ( 1944.0,   923.6,   191.8,   -12.8, -1599.3, -2398.4, -2802.9, -1996.6, -1597.9, -1039.4,
       -1061.3, -1345.8,   -62.4,   -25.9,   -21.1,    -9.9,    79.4,   120.0,    65.3,    30.5,
          -1.0,  -278.2,  -994.7, -1794.1, -4315.4, -4438.8, -4437.1, -4443.0, -4378.4, -3605.0,
       -3977.6, -4239.2, -2402.8, ),
      (  459.9,   210.9,    -9.9,  -588.9, -1913.3, -2591.4, -3053.4, -2975.4, -2016.1, -1992.1,
       -1619.8, -1926.1,   -38.2,   -23.6,   -23.0,    -0.1,    30.0,    30.2,    24.8,    15.0,
          10.0,     5.0,  -262.5, -1178.6, -2949.8, -4621.6, -4421.9, -4450.7, -4401.8, -4361.9,
       -4401.3, -4437.3, -3876.6, ),
      (  108.9,    23.3,   -97.1, -1136.2, -2098.3, -2602.7, -3192.4, -3178.5, -2608.1, -2588.1,
       -2597.9,  -660.0,    -9.9,   -40.1,   -29.0,   -10.1,     5.0,    10.0,    10.0,    10.0,
          10.0,    10.0,     0.0,  -276.1, -1413.3, -4101.5, -3835.0, -4424.2, -4415.9, -4443.6,
       -4443.1, -4460.6, -4399.8, ),
      (   69.7,   -10.0,  -402.6, -1785.3, -2391.5, -2800.2, -3278.6, -3408.0, -3111.9, -3002.1,
       -3014.9,  -782.6,   -44.6,   -17.9,   -21.9,   -29.0,   -21.0,   -14.0,   -12.0,     0.0,
           0.0,   -10.0,   -10.0,  -103.2, -1775.1, -2001.9, -2965.9, -3711.9, -3032.2, -2801.0,
          -3.1,  -920.6, -1273.1, ),
      (   64.0,   -10.0, -1313.9, -2190.4, -2769.9, -2899.3, -3317.6, -3536.4, -3564.0, -3364.1,
       -3331.0, -1715.7,   -13.2,   -47.9,   -19.7,   -46.2,   -45.9,   -40.0,   -33.1,    -3.1,
         -24.9,    -4.2,   -10.0,  -202.8, -1409.1, -1982.8,  -298.8,     0.0,  -575.5,  -339.9,
          -3.9,   -10.0,   -10.0, ),
      (   -8.5,   -10.0, -1714.1, -2321.6, -2790.2, -3072.8, -3446.8, -3621.8, -3639.0, -3635.0,
       -3581.5, -3569.9, -3367.2, -1055.4,  -227.7,  -117.0,   -53.2,   -47.7,   -61.2,   -55.2,
         -53.0,   -50.0,  -133.0,  -507.9, -1065.2, -1840.9, -2201.4,  -907.3,   192.0,    32.1,
          -2.2,   -10.0,   -10.0, ),
      (    7.2,   -45.9, -1526.8, -2196.7, -2588.5, -3178.9, -3593.6, -3630.1, -3623.9, -3628.0,
       -3621.0, -3619.0, -3709.8, -3908.8, -3558.9, -1507.3,  -237.2,   -45.2,    -4.3,   -62.2,
         -48.1,   -32.1,  -394.2,  -807.6, -1193.3, -1139.0, -2075.7, -2395.8, -2128.4, -1034.6,
          42.3,   112.0,   125.4, ),
      (  134.6,   -39.1, -1078.0, -2179.4, -2595.1, -3002.4, -3565.4, -3618.1, -3666.8, -3679.0,
       -3648.2, -3598.0, -3597.9, -3602.0, -3695.7, -3695.5, -1106.2,  -725.6,   -12.7,   -10.0,
         -92.3,  -151.6,  -591.2,  -977.7, -2939.2, -1435.4, -3011.0, -2948.8, -2399.3, -2182.2,
       -1800.0, -1789.0, -1570.9, ),
      (   55.0,   -35.2,  -838.4, -1609.6, -2488.7, -3268.8, -3594.0, -3613.1, -3724.5, -3727.1,
       -3714.9, -3671.4, -3739.6, -3718.0, -3655.8, -3607.0, -3826.2, -3652.7, -3069.2,  -352.8,
        -181.9,  -396.3,  -994.3, -1254.2, -3228.1, -3401.8, -3468.6, -2829.9, -2184.4, -1017.2,
       -1006.6,  -808.0, -1059.1, ),
      (   -8.7,   -18.7,  -141.0, -1008.6, -2264.6, -3205.6, -3619.4, -3620.2, -3654.8, -3655.1,
       -3631.0, -3622.0, -3611.0, -3609.0, -3616.1, -3611.1, -3600.8, -3851.1, -3622.7, -2515.4,
       -1578.2, -1492.1, -1769.6, -3051.5, -3383.7, -3415.2, -3402.4, -3268.5, -1954.1,  -192.3,
         -40.4,   -13.9,   -34.0, ),
      (    2.1,   -10.0,   -65.5,  -487.2, -1596.1, -2703.7, -3579.7, -3627.0, -3628.0, -3612.1,
       -3609.9, -3578.6, -3542.4, -3511.9, -3587.7, -3583.2, -3545.1, -3585.9, -3544.5, -3302.6,
       -3388.8, -3189.2, -3241.8, -3287.6, -3321.9, -3363.9, -3249.2, -2189.4,  -127.6,   -33.0,
         -43.8,   -35.9,   -10.0, ),
      (   17.1,     2.0,   -25.7,  -129.9, -1088.0, -1432.3, -3386.1, -3450.0, -3524.9, -3534.9,
       -3478.5, -3322.9, -3172.3, -3173.0, -3485.8, -3474.6, -3377.6, -3338.1, -3346.9, -3323.2,
       -3332.8, -3348.0, -3326.2, -3248.5, -3205.8, -3292.8, -3125.2,  -576.5,   -10.0,   -95.3,
         -54.2,   -34.1,   -10.1, ),
      (   17.2,     5.1,   -26.6,   -16.6,  -984.4, -1684.9, -2579.5, -3005.0, -3172.7, -3123.2,
       -2726.5, -2209.1, -2458.3, -2407.9, -3268.3, -3300.9, -3207.9, -3135.3, -3060.5, -2991.1,
       -3029.2, -3116.5, -3162.7, -3184.2, -3205.8, -3252.9, -2952.0,  -322.8,   -16.8,  -100.2,
         -44.2,   -30.0,   -12.1, ),
      (   13.1,   -10.0,   -29.0,   -98.8, -1000.0, -1589.9, -1674.0, -1989.6, -1577.0, -1851.7,
       -1797.0, -1904.0, -1999.0, -2290.0, -2206.1, -2795.8, -2937.9, -2917.3, -2723.1, -2597.1,
       -2752.5, -2909.1, -2999.6, -3094.5, -3177.6, -3381.8, -2460.4,  -200.7,   -85.4,   -73.2,
         -36.1,   -16.1,     0.9, ),
      (   14.0,    -9.7,   -41.8,  -140.8,  -789.2, -1386.5, -1399.6, -1211.4, -1293.0, -1023.3,
       -1212.6, -1503.1, -1733.2, -1991.2, -1894.8, -1864.4, -2391.9, -2511.8, -2282.3, -2586.4,
       -2724.9, -2879.5, -2959.6, -3129.6, -3203.8, -3205.9,  -817.6,  -102.9,   -89.2,   -50.3,
         -33.1,    -9.1,     7.8, ),
      (   19.3,    -9.6,   -28.7,   -67.7,  -205.8,  -796.4,  -994.9,  -950.1,  -881.5,  -601.2,
        -796.9,  -869.5,  -814.3, -1010.9, -1083.3, -1199.0, -1203.9, -1873.3, -1800.9, -2018.7,
       -2598.2, -2898.5, -3047.9, -3258.1, -3224.9, -1994.5,  -406.1,  -116.8,   -65.2,   -39.1,
         -17.2,    13.8,    20.2, ),
      (   41.6,    14.1,    -5.8,   -28.8,   -43.9,   -61.6,   -58.8,   -89.9,   -80.0,  -100.0,
        -103.2,  -105.9,  -134.1,  -203.9,  -211.7,  -228.9,  -537.9, -1026.6, -1397.3, -2180.6,
       -2476.4, -2800.7, -2668.6, -2293.7, -1026.3,  -600.2,  -230.9,   -80.6,   -44.1,   -29.0,
          -9.1,     8.9,    38.4, ),
      (  105.0,    44.1,     7.2,     0.0,    -7.9,   -27.9,   -31.0,   -28.0,   -41.9,   -47.0,
         -47.0,   -49.0,   -49.0,   -39.0,   -29.0,   -37.9,  -196.6,  -395.4,  -885.8, -1677.7,
       -2276.7, -1953.6,  -873.2,  -570.8,  -303.7,  -200.0,  -100.1,   -50.1,   -33.1,   -23.1,
         -10.1,    36.6,    35.8, ),
      (   85.3,    81.1,    31.2,    15.0,    11.1,     2.1,    -7.0,   -17.9,   -19.0,   -20.0,
         -22.0,   -25.0,   -18.0,   -10.0,    -3.0,    -8.0,   -24.8,   -11.4,   -77.7,  -502.6,
       -1330.0, -1583.2,  -704.2,  -377.0,  -219.5,   -99.1,   -42.1,   -34.1,   -28.0,   -14.1,
          -4.1,    16.4,    25.0, ),
      (  189.7,    83.3,    79.0,    45.2,    29.9,    15.0,   -10.7,    -1.3,   -12.0,    -9.0,
         -12.0,    -9.0,    -1.0,    -1.0,     1.0,     1.0,    -0.3,    -9.0,   -13.9,   -42.7,
         -51.4,  -109.8,  -397.0,  -187.9,   -58.6,   -20.2,    -9.0,   -20.1,   -19.0,    -4.2,
          11.8,    24.0,    36.4, ),
      (  210.0,   148.8,   104.6,    76.4,    56.2,    26.2,    14.1,     8.1,     1.1,     2.9,
           5.0,     3.1,     6.0,     1.0,     3.0,     3.0,    -9.6,    -0.3,   -10.0,   -24.8,
         -28.1,   -28.0,   -95.3,   -58.0,   -31.1,     4.8,     8.9,     0.1,    -1.0,    16.0,
          12.0,    32.1,    24.7, ),
      (  259.9,   163.1,   111.3,    66.4,   105.2,    97.8,    20.5,    30.1,    20.2,    17.1,
          12.0,    14.0,     8.0,     7.0,     9.0,     6.1,    15.9,    20.1,     9.0,    14.8,
          -1.0,     4.2,    -0.8,     3.8,    17.9,    46.0,    53.6,    21.5,    39.1,    32.4,
          34.1,    31.9,    19.2, ),
      (  272.2,   149.6,   114.9,   130.2,    88.9,    55.0,    95.1,    92.3,    93.1,    58.1,
          52.1,    23.1,    11.9,    71.1,    66.6,    91.5,    70.5,    76.5,    59.7,    54.1,
          11.9,    68.2,    64.6,    76.9,    37.2,    37.3,    44.9,    63.2,    76.9,    67.6,
          51.2,    33.1,     9.2, ),
)

def __createExampleCurvData():
    '''
    Creates and returns example longitude, latitudes, depth, and data
    for a curvilinear grid.  Assigns grid center point data[i, j, k]
        = sin(lon[i, j, k]) * cos(lat[i, j, k]) / log(depth[i, j, k] + 1.0)
            for defined areas
        = 1.0E20 for undefined areas

    Arguments:
        None
    Returns:
        (corner_lons, corner_lats, corner_depths,
         center_lons, center_lats, center_depths, data) where:
        corner_lons:   numpy.float64 3D array of curvilinear corner point longitude coordinates
        corner_lats:   numpy.float64 3D array of curvilinear corner point latitude coordinates
        corner_depths: numpy.float64 3D array of curvilinear corner point depth coordinates
        center_lons:   numpy.float64 3D array of curvilinear center point longitude coordinates
        center_lats:   numpy.float64 3D array of curvilinear center point latitude coordinates
        center_depths: numpy.float64 3D array of curvilinear center point depth coordinates
        data:          numpy.float64 3D array of curvilinear center point data values
    '''
    corner_lons = None
    corner_lats = None
    corner_depths = None
    center_lons = None
    center_lats = None
    center_depths = None
    data = None
    
    return (corner_lons, corner_lats, corner_depths, center_lons, center_lats, center_depths, data)


def __createExampleRectData():
    '''
    Creates and returns example longitude, latitudes, depth, and data
    for a rectilinear grid.  Covers approximately the same region given
    by __createExampleCurvData.  Assigns grid center point data[i, j, k]
        = sin(lon[i]) * cos(lat[j]) / log(depth[k] + 1.0)
            for defined areas
        = 1.0E34 for undefined areas

    Arguments:
        None
    Returns:
        (lon_edges, lat_edges, depth_edges, data) where:
        lon_edges:   numpy.float64 1D array of rectilinear edge longitudes
        lat_edges:   numpy.float64 1D array of rectilinear edge latitudes
        depth_edges: numpy.float64 1D array of rectilinear edge depths
        data:        numpy.float64 2D array of rectilinear center point data values
    '''
    lon_edges = None
    lat_edges = None
    depth_edges = None
    data = None

    return (lon_edges, lat_edges, depth_edges, data)


def __printDiffs(grid_lons, grid_lats, grid_depths, undef_val, expect_data, found_data):
    '''
    Prints significant differences between expect_data and found_data
    along with the location of these differences

    Arguments:
        grid_lons:   numpy 3D array of grid longitudes
        grid_lats:   numpy 3D array of grid latitudes
        grid_depths: numpy 3D array of grid depths
        undef_val:   numpy array of one value; the undefined data value
        expect_data: numpy 3D array of expected data values
        found_data:  numpy 3D array of data values to check
    Returns:
        None
    Raises:
        ValueError:  if the array shapes do not match
    '''
    if (len(grid_lons.shape) != 3):
        raise ValueError("grid_lons is not 3D")
    if (grid_lats.shape != grid_lons.shape):
        raise ValueError("grid_lats.shape != grid_lons.shape")
    if (grid_depths.shape != grid_lons.shape):
        raise ValueError("grid_depth.shape != grid_lons.shape")
    if (expect_data.shape != grid_lons.shape):
        raise ValueError("expect_data.shape != grid_lons.shape")
    if (found_data.shape != grid_lons.shape):
        raise ValueError("found_data.shape != grid_lons.shape")
    different = (numpy.abs(expect_data - found_data) > 0.05)
    diff_lons = grid_lons[different]
    diff_lats = grid_lats[different]
    diff_depths = grid_depths[different]
    diff_expect = expect_data[different]
    diff_found = found_data[different]
    diff_list = [ ]
    for (lon, lat, depth, expect, found) in \
            zip(diff_lons, diff_lats, diff_depths, diff_expect, diff_found):
        if expect == undef_val:
            # most serious - should have been masked out
            diff_list.append([2, lon, lat, depth, expect, found])
        elif found == undef_val:
            # least serious - destination not covered by source
            diff_list.append([0, lon, lat, depth, expect, found])
        else:
            # might be of concern
            diff_list.append([1, lon, lat, depth, expect, found])
    # order primarily from least to most serious, 
    # secondarily smallest to largest longitude,
    # tertiarily smallest to largest latitude
    diff_list.sort()
    num_not_undef = 0
    num_undef = 0
    num_diff = 0
    for (_, lon, lat, depth, expect, found) in diff_list:
        if expect == undef_val:
            num_not_undef += 1
            print "lon = %#7.3f, lat = %7.3f, depth = %7.1f, expect =  undef, " \
                  "found = %#6.3f" % (lon, lat, depth, found)
        elif found == undef_val:
            num_undef += 1
            print "lon = %#7.3f, lat = %7.3f, depth = %7.1f, expect = %#6.3f, " \
                  "found =  undef" % (lon, lat, depth, expect)
        else:
            num_diff += 1
            print "lon = %#7.3f, lat = %7.3f, depth = %7.1f, expect = %#6.3f, " \
                  "found = %#6.3f, diff = %#6.3f" \
                  % (lon, lat, depth, expect, found, found - expect)
    print "%3d undefined when defined expected" % num_undef
    print "%3d with absolute difference > 0.05" % num_diff
    print "%3d defined when undefined expected" % num_not_undef
    print "%3d values in the grid" \
            % (expect_data.shape[0] * expect_data.shape[1] * expect_data.shape[2])


if __name__ == '__main__':
    try:
        while True:
            print 'cw2r: curvilinear with corners to rectilinear'
            print 'co2r: curvilinear without corners to rectilinear'
            print 'r2cw: rectilinear to curvilinear with corners'
            print 'r2co: rectilinear to curvilinear without corners'
            print 'Ctrl-D to quit'
            direction = raw_input('Regrid test to run? ')
            direction = direction.strip().lower()
            if direction in ('cw2r', 'co2r', 'r2cw', 'r2co'):
                break
    except EOFError:
        raise SystemExit(0)

    # Synthesize test data
    (curv_corner_lons, curv_corner_lats, curv_corner_depths,
     curv_center_lons, curv_center_lats, curv_center_depths, curv_data) = __createExampleCurvData()
    curv_center_ignore = (curv_data >= 256.0)
    (rect_lon_edges, rect_lat_edges, rect_depth_edges, rect_data) = __createExampleRectData()
    rect_center_ignore = (rect_data >= 256.0)
    undef_val = numpy.array([-1.0E10], dtype=numpy.float64)

    # Create the expected results on the curvilinear grid
    curv_expect_data = curv_data.copy()
    curv_expect_data[curv_center_ignore] = undef_val

    # Create the expected results on the rectilinear grid
    rect_expect_data = rect_data.copy()
    rect_expect_data[rect_center_ignore] = undef_val

    # Generate the 2D rectilinear longitudes and latitudes arrays only to 
    # simplify printing differences; not used for generating rectilinear grids
    rect_mid_lons = 0.5 * (rect_lon_edges[:-1] + rect_lon_edges[1:])
    rect_mid_lats = 0.5 * (rect_lat_edges[:-1] + rect_lat_edges[1:])
    rect_mid_depths = 0.5 * (rect_depth_edges[:-1] + rect_depth_edges[1:])
    rect_center_shape = (rect_mid_lons.shape[0], rect_mid_lats.shape[0], rect_mid_depths.shape[0])
    # Creating these in C order
    rect_center_lons = numpy.repeat(rect_mid_lons, rect_center_shape[1] * rect_center_shape[2]) \
                            .reshape(rect_center_shape)
    rect_center_lats = numpy.repeat(numpy.tile(rect_mid_lats, rect_center_shape[0]), rect_center_shape[2]) \
                            .reshape(rect_center_shape)
    rect_center_depths = numpy.tile(rect_mid_depths, rect_center_shape[0] * rect_center_shape[1]) \
                              .reshape(rect_center_shape)

    # Initialize ESMP
    ESMP.ESMP_Initialize()

    # Create the regridder
    regridder = CurvRect3DRegridder()

    if direction in ('cw2r', 'r2cw'):
        # Create the curvilinear grid with corner and center points
        regridder.createCurvGrid(curv_center_lons, curv_center_lats, curv_center_depths,
                                 True, curv_center_ignore, curv_corner_lons, curv_corner_lats)
    elif direction in ('co2r', 'r2co'):
        # Create the curvilinear grid with only center points
        regridder.createCurvGrid(curv_center_lons, curv_center_lats, curv_center_depths,
                                 True, curv_center_ignore)
    else:
        raise ValueError("unexpected direction of %s" % direction)

    # Create the rectilinear grid with corner and center points
    regridder.createRectGrid(rect_lon_edges, rect_lat_edges, rect_depth_edges, 
                             True, rect_center_ignore)

    if direction in ('cw2r', 'co2r'):
        print ""
        if direction == 'cw2r':
            print "Examining rectilinear results from curvilinear with corners"
        else:
            print "Examining rectilinear results from curvilinear without corners"

        # Create the curvilinear source field
        regridder.assignCurvField(curv_data)

        # Create the rectilinear destination field
        regridder.assignRectField()

        # Regrid from curvilinear to rectilinear using the bilinear method
        rect_regrid_data = regridder.regridCurvToRect(undef_val, ESMP.ESMP_REGRIDMETHOD_BILINEAR)

        # Print the differences between the expected and regrid data
        print ""
        print "analytic (expect) versus bilinear regridded (found) differences"
        __printDiffs(rect_center_lons, rect_center_lats, rect_center_depths, 
                     undef_val, rect_expect_data, rect_regrid_data)

        # Regrid from curvilinear to rectilinear using the patch method
        rect_regrid_data = regridder.regridCurvToRect(undef_val, ESMP.ESMP_REGRIDMETHOD_PATCH)

        # Print the differences between the expected and regrid data
        print ""
        print "analytic (expect) versus patch regridded (found) differences"
        __printDiffs(rect_center_lons, rect_center_lats, rect_center_depths,
                     undef_val, rect_expect_data, rect_regrid_data)

        if direction == 'cw2r':
            # Regrid from curvilinear to rectilinear using the conserve method
            # Corners required for this method
            rect_regrid_data = regridder.regridCurvToRect(undef_val, ESMP.ESMP_REGRIDMETHOD_CONSERVE)

            # Print the differences between the expected and regrid data
            print ""
            print "analytic (expect) versus conserve regridded (found) differences"
            __printDiffs(rect_center_lons, rect_center_lats, rect_center_depths,
                         undef_val, rect_expect_data, rect_regrid_data)

    elif direction in ('r2cw', 'r2co'):
        print ""
        if direction == 'r2cw':
            print "Examining curvilinear with corners results from rectilinear"
        else:
            print "Examining curvilinear without corners results from rectilinear"

        # Create the rectilinear source field
        regridder.assignRectField(rect_data)

        # Create the curvilinear destination field
        regridder.assignCurvField(None)

        # Regrid from rectilinear to curvilinear using the bilinear method
        curv_regrid_data = regridder.regridRectToCurv(undef_val, ESMP.ESMP_REGRIDMETHOD_BILINEAR)

        # Print the differences between the expected and regrid data
        print ""
        print "analytic (expect) versus bilinear regridded (found) differences"
        __printDiffs(curv_center_lons, curv_center_lats, curv_center_depths,
                     undef_val, curv_expect_data, curv_regrid_data)

        # Regrid from rectilinear to curvilinear using the patch method
        curv_regrid_data = regridder.regridRectToCurv(undef_val, ESMP.ESMP_REGRIDMETHOD_PATCH)

        # Print the differences between the expected and regrid data
        print ""
        print "analytic (expect) versus patch regridded (found) differences"
        __printDiffs(curv_center_lons, curv_center_lats, curv_center_depths,
                     undef_val, curv_expect_data, curv_regrid_data)

        if direction == 'r2cw':
            # Regrid from rectilinear to curvilinear using the conserve method
            # Corners required for this method
            curv_regrid_data = regridder.regridRectToCurv(undef_val, ESMP.ESMP_REGRIDMETHOD_CONSERVE)

            # Print the differences between the expected and regrid data
            print ""
            print "analytic (expect) versus conserve regridded (found) differences"
            __printDiffs(curv_center_lons, curv_center_lats, curv_center_depths,
                         undef_val, curv_expect_data, curv_regrid_data)

    else:
        raise ValueError("unexpected direction of %s" % direction)

    # Done with this regridder
    regridder.finalize()

    # Done with ESMP    
    ESMP.ESMP_Finalize()


