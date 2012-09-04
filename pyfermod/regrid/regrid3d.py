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

    See the ESMPControl singleton class to simplify initializing and
    finalizing ESMP once, and only once, for a Python session.
    '''


    def __init__(self, earth_radius=6371.007, debug=False):
        '''
        Initializes to an empty regridder.  The ESMP module must be
        imported and ESMP.ESMP_Initialize() called (possibly through
        invoking ESMPControl().startCheckESMP()) prior to calling
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
        # Radius of the earth, in kilometers, to use in regridding
        self.__earth_rad = earth_radius
        self.__debug = debug
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
                       center_ignore=None, levels_are_depths=True,
                       corner_lons=None, corner_lats=None,
                       corner_levs=None, corner_ignore=None):
        '''
        Create the curvilinear grid as an ESMP_Grid using the provided center
        longitudes, latitudes, and depths (or elevations) as the grid
        center points, and, if given, the grid corner longitudes, latitudes,
        and depths (or elevations) as the grid corner points.  Curvilinear
        data is assigned to the center points.  Grid point coordinate
        coord[i,j,k] is assigned from lon[i,j,k], lat[i,j,k], and levs[i,j,k].

        Center point [i,j,k] is taken to be the center point of the
        quadrilaterally-faced hexahedron defined by the corner points:
        corner_pt[i, j, k], corner_pt[i+1, j, k], corner_pt[i+1, j+1, k],
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
            center_ignore:     3D array of boolean-like values, indicating
                               if the corresponding grid center point should
                               be ignored in the regridding; if None, no
                               grid center points will be ignored
            levels_are_depths: True -  levels are depths and subtracted from
                                       the earth radius;
                               False - levels are elevations and added to
                                       the earth radius
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
            ValueError: if the shape (dimensionality) of an argument is
                        invalid, or if a value in an argument is invalid
            TypeError:  if an argument is not array-like
        '''
        # Make sure center_lons is an appropriate array-like argument
        center_lons_array = numpy.array(center_lons, dtype=numpy.float64, copy=False)
        if len(center_lons_array.shape) != 3:
            raise ValueError("center_lons must be three-dimensional")

        # Make sure center_lats is an appropriate array-like argument
        center_lats_array = numpy.array(center_lats, dtype=numpy.float64, copy=False)
        if center_lats_array.shape != center_lons_array.shape:
            raise ValueError("center_lats and center_lons must have the same shape")

        # Make sure center_lats is an appropriate array-like argument
        center_levs_array = numpy.array(center_levs, dtype=numpy.float64, copy=False)
        if center_levs_array.shape != center_lons_array.shape:
            raise ValueError("center_levs and center_lons must have the same shape")

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

        # Mask for X,Y,Z coordinates to compute
        if center_ignore_array != None:
            good_centers = numpy.logical_not(center_ignore_array).flatten('F')
        else:
            good_centers = numpy.array([True] * (center_lons_array.shape[0] * \
                                                 center_lons_array.shape[1] * \
                                                 center_lons_array.shape[2]) )

        if (corner_lons != None) and (corner_lats != None) and (corner_levs != None):
            # Corner points specified

            # Make sure corner_lons is an appropriate array-like argument
            corner_lons_array = numpy.array(corner_lons, dtype=numpy.float64, copy=False)
            if corner_lons_array.shape != (center_lons_array.shape[0] + 1,
                                           center_lons_array.shape[1] + 1,
                                           center_lons_array.shape[2] + 1):
                raise ValueError("corner_lons must have one more point along " \
                                 "each dimension when compared to center_lons")

            # Make sure corner_lats is an appropriate array-like argument
            corner_lats_array = numpy.array(corner_lats, dtype=numpy.float64, copy=False)
            if corner_lats_array.shape != corner_lons_array.shape:
                raise ValueError("corner_lats and corner_lons must have the same shape")

            # Make sure corner_lats is an appropriate array-like argument
            corner_levs_array = numpy.array(corner_levs, dtype=numpy.float64, copy=False)
            if corner_levs_array.shape != corner_lons_array.shape:
                raise ValueError("corner_levs and corner_lons must have the same shape")

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

            # Mask for X,Y,Z coordinates to compute
            if corner_ignore_array != None:
                good_corners = numpy.logical_not(corner_ignore_array).flatten('F')
            else:
                good_corners = numpy.array([True] * (corner_lons_array.shape[0] * \
                                                     corner_lons_array.shape[1] * \
                                                     corner_lons_array.shape[2]) )

        elif (corner_lons != None) or (corner_lats != None) or (corner_levs != None):
            raise ValueError("one or two, but not all three, of corner_lons, " \
                             "corner_lats, and corner_levs are given")
        elif corner_ignore != None:
            raise ValueError("corner_ignore given without corner_lons, " \
                             "corner_lats, and corner_levs")
        else:
            # No corner points specified
            corner_lons_array = None
            corner_lats_array = None
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
        # using ESMP_GridCreateNoPeriDim for the typical case (not
        # the whole world) in Ferret.
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
            lons = corner_lons_array.flatten('F')
            lons = numpy.deg2rad(lons[good_corners])
            lats = corner_lats_array.flatten('F')
            lats = numpy.deg2rad(lats[good_corners])
            levs = corner_levs_array.flatten('F')
            levs = levs[good_corners] / 1000.0
            if levels_are_depths:
                levs *= -1.0
            levs += self.__earth_rad
            levs /= self.__earth_rad
            # XY plane through the prime meridian, Z toward central atlantic
            # 0 lon, 0 lat = (0, 0, R); 90 lon, 0 lat = (R, 0, 0); * lon, 90 lat = (0, R, 0)
            cos_lats = numpy.cos(lats)
            grid_x_coords[:] = 0.0
            grid_x_coords[good_corners] = levs * numpy.sin(lons) * cos_lats
            grid_y_coords[:] = 0.0
            grid_y_coords[good_corners] = levs * numpy.sin(lats)
            grid_z_coords[:] = 0.0
            grid_z_coords[good_corners] = levs * numpy.cos(lons) * cos_lats

            if self.__debug:
                fout = open("curv_corner_xyz.txt", "w")
                try:
                    print >>fout, "curv_corner_x = %s" % \
                        self.__myArrayStr(grid_x_coords, corner_lons_array.shape)
                    print >>fout, "curv_corner_y = %s" % \
                        self.__myArrayStr(grid_y_coords, corner_lons_array.shape)
                    print >>fout, "curv_corner_z = %s" % \
                        self.__myArrayStr(grid_z_coords, corner_lons_array.shape)
                finally:
                    fout.close()
            
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
        lons = center_lons_array.flatten('F')
        lons = numpy.deg2rad(lons[good_centers])
        lats = center_lats_array.flatten('F')
        lats = numpy.deg2rad(lats[good_centers])
        levs = center_levs_array.flatten('F')
        levs = levs[good_centers] / 1000.0
        if levels_are_depths:
            levs *= -1.0
        levs += self.__earth_rad
        levs /= self.__earth_rad
        # XY plane through the prime meridian, Z toward central atlantic
        # 0 lon, 0 lat = (0, 0, R); 90 lon, 0 lat = (R, 0, 0); * lon, 90 lat = (0, R, 0)
        cos_lats = numpy.cos(lats)
        grid_x_coords[:] = 0.0
        grid_x_coords[good_centers] = levs * numpy.sin(lons) * cos_lats
        grid_y_coords[:] = 0.0
        grid_y_coords[good_centers] = levs * numpy.sin(lats)
        grid_z_coords[:] = 0.0
        grid_z_coords[good_centers] = levs * numpy.cos(lons) * cos_lats

        if self.__debug:
            fout = open("curv_center_xyz.txt", "w")
            try:
                print >>fout, "curv_center_x = %s" % \
                    self.__myArrayStr(grid_x_coords, self.__curv_shape)
                print >>fout, "curv_center_y = %s" % \
                    self.__myArrayStr(grid_y_coords, self.__curv_shape)
                print >>fout, "curv_center_z = %s" % \
                    self.__myArrayStr(grid_z_coords, self.__curv_shape)
            finally:
                fout.close()

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

            # Assign the data to the field array in the ESMP_Field;
            # flatten in the same order as the grid coordinates
            field_ptr[:] = data_array.flatten('F')

            if self.__debug:
                fout = open("curv_data.txt", "w")
                try:
                    print >>fout, "curv_data = %s" % \
                        self.__myArrayStr(field_ptr, self.__curv_shape)
                finally:
                    fout.close()


    def createRectGrid(self, center_lons, center_lats, center_levs,
                       center_ignore=None, levels_are_depths=True,
                       corner_lons=None, corner_lats=None, corner_levs=None,
                       corner_ignore=None):
        '''
        Create the rectilinear grid as an ESMP_Grid using the provided cell
        center and edge longitudes, latitudes, and levels to define the grid
        center and corner points.  Rectilinear data is assigned to the center
        points.  Grid point [i,j,k] is assigned from lons[i], lats[j], and
        levs[k].

        Center point [i,j,k] is taken to be the center point of the
        quadrilaterally-faced hexahedron defined by the corner points:
        corner_pt[i, j, k], corner_pt[i+1, j, k], corner_pt[i+1, j+1, k],
        corner_pt([i, j+1, k], corner_pt[i, j, k+1], corner_pt[i+1, j, k+1],
        corner_pt[i+1, j+1, k+1], and corner_pt[i, j+1, k+1].

        Any previous ESMP_Grid, ESMP_Field, or ESMP regridding procedures
        are destroyed.

        Arguments:
            center_lons:       1D array of longitudes, in degrees,
                               for each of the rectilinear grid cells center
                               planes
            center_lats:       1D array of latitudes, in degrees,
                               for each of the rectilinear grid cells center
                               planes
            center_levs:       1D array of levels, in meters from the earth
                               radius, for each of the rectilinear grid cells
                               center planes
            center_ignore:     3D array of boolean-like values, indicating if
                               the corresponding grid center point should be
                               ignored in the regridding; if None, no grid
                               center points will be ignored
            levels_are_depths: True -  levels are depths and subtracted from
                                       the earth radius,
                               False - levels are elevations and added to
                                       the earth radius
            corner_lons:       1D array of longitudes, in degrees,
                               for each of the rectilinear grid cells corner
                               planes
            corner_lats:       1D array of latitudes, in degrees,
                               for each of the rectilinear grid cells corner
                               planes
            corner_levs:       1D array of levels, in meters from the earth
                               radius, for each of the rectilinear grid cells
                               corner planes
            corner_ignore:     3D array of boolean-like values, indicating if
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

        # Make sure center_levs is an appropriate array-like argument
        center_levs_array = numpy.array(center_levs, dtype=numpy.float64, copy=True)
        if len(center_levs_array.shape) != 1:
            raise ValueError("center_levs must be one-dimensional")

        if center_ignore == None:
            # Using all center points, no mask created
            center_ignore_array = None
        else:
            # Make sure center_ignore is an appropriate array-like argument
            center_ignore_array = numpy.array(center_ignore, dtype=numpy.bool, copy=False)
            if center_ignore_array.shape != (center_lons_array.shape[0],
                                             center_lats_array.shape[0],
                                             center_levs_array.shape[0]):
                raise ValueError("center_ignore must have the shape " \
                                 "( len(center_lons), len(center_lats), len(center_levs) )")
            # If not actually ignoring any center points, do not create a mask
            if not center_ignore_array.any():
                center_ignore_array = None

        if (corner_lons != None) and (corner_lats != None) and (corner_levs != None):
            # Corner points specified

            # Make sure corner_lons is an appropriate array-like argument
            corner_lons_array = numpy.array(corner_lons, dtype=numpy.float64, copy=False)
            if corner_lons_array.shape != (center_lons_array.shape[0] + 1, ):
                raise ValueError("corner_lons must have shape ( len(center_lons) + 1, )")

            # Make sure corner_lats is an appropriate array-like argument
            corner_lats_array = numpy.array(corner_lats, dtype=numpy.float64, copy=True)
            if corner_lats_array.shape != (center_lats_array.shape[0] + 1, ):
                raise ValueError("corner_lats must have shape ( len(center_lats) + 1, )")

            # Make sure corner_levs is an appropriate array-like argument
            corner_levs_array = numpy.array(corner_levs, dtype=numpy.float64, copy=True)
            if corner_levs_array.shape != (center_levs_array.shape[0] + 1, ):
                raise ValueError("corner_levs must have shape ( len(center_levs) + 1, )")

            if corner_ignore == None:
                # Using all corner points, no mask created
                corner_ignore_array = None
            else:
                # Make sure corner_ignore is an appropriate array-like argument
                corner_ignore_array = numpy.array(corner_ignore, dtype=numpy.bool, copy=False)
                if corner_ignore_array.shape != (corner_lons_array.shape[0],
                                                 corner_lats_array.shape[0],
                                                 corner_levs_array.shape[0]):
                    raise ValueError("corner_ignore must have the shape " \
                                     "( len(corner_lons), len(corner_lats), len(corner_levs) )")
                # If not actually ignoring any points, do not create a mask
                if not corner_ignore_array.any():
                    corner_ignore_array = None

        elif (corner_lons != None) or (corner_lats != None) or (corner_levs != None):
            raise ValueError("one or two, but not all three, of corner_lons, " \
                             "corner_lats, and corner_levs are given")
        elif corner_ignore != None:
            raise ValueError("corner_ignore given without corner_lons, " \
                             "corner_lats, and corner_levs")
        else:
            # No corner points specified
            corner_lons_array = None
            corner_lats_array = None
            corner_levs_array = None
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
        # using ESMP_GridCreateNoPeriDim for the typical case (not
        # the whole world) in Ferret.
        self.__rect_shape = (center_lons_array.shape[0],
                             center_lats_array.shape[0],
                             center_levs_array.shape[0])
        grid_shape = numpy.array(self.__rect_shape, dtype=numpy.int32)
        self.__rect_grid = ESMP.ESMP_GridCreateNoPeriDim(grid_shape,
                                         ESMP.ESMP_COORDSYS_CART,
                                         ESMP.ESMP_TYPEKIND_R8)

        if corner_lons_array != None:
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
            corner_shape = (corner_lons_array.shape[0],
                            corner_lats_array.shape[0],
                            corner_levs_array.shape[0])
            lons = numpy.deg2rad(corner_lons_array)
            cos_lons = numpy.tile(numpy.cos(lons),
                                  corner_shape[1] * corner_shape[2])
            sin_lons = numpy.tile(numpy.sin(lons),
                                  corner_shape[1] * corner_shape[2])
            lats = numpy.deg2rad(corner_lats_array)
            cos_lats = numpy.tile(numpy.repeat(numpy.cos(lats), corner_shape[0]),
                                  corner_shape[2])
            sin_lats = numpy.tile(numpy.repeat(numpy.sin(lats), corner_shape[0]),
                                  corner_shape[2])
            levs = corner_levs_array / 1000.0
            if levels_are_depths:
                levs *= -1.0
            levs += self.__earth_rad
            levs /= self.__earth_rad
            levs = numpy.repeat(levs, corner_shape[0] * corner_shape[1])
            # XY plane through the prime meridian, Z toward central atlantic
            # 0 lon, 0 lat = (0, 0, R); 90 lon, 0 lat = (R, 0, 0); * lon, 90 lat = (0, R, 0)
            grid_x_coords[:] = levs * sin_lons * cos_lats
            grid_y_coords[:] = levs * sin_lats
            grid_z_coords[:] = levs * cos_lons * cos_lats

            if self.__debug:
                fout = open("rect_corner_xyz.txt", "w")
                try:
                    print >>fout, "rect_corner_x = %s" % \
                        self.__myArrayStr(grid_x_coords, corner_shape)
                    print >>fout, "rect_corner_y = %s" % \
                        self.__myArrayStr(grid_y_coords, corner_shape)
                    print >>fout, "rect_corner_z = %s" % \
                        self.__myArrayStr(grid_z_coords, corner_shape)
                finally:
                    fout.close()

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

        # Retrieve the grid center coordinate arrays in the ESMP_Grid
        grid_x_coords = ESMP.ESMP_GridGetCoordPtr(self.__rect_grid, 0,
                                                  ESMP.ESMP_STAGGERLOC_CENTER_VCENTER)
        grid_y_coords = ESMP.ESMP_GridGetCoordPtr(self.__rect_grid, 1,
                                                  ESMP.ESMP_STAGGERLOC_CENTER_VCENTER)
        grid_z_coords = ESMP.ESMP_GridGetCoordPtr(self.__rect_grid, 2,
                                                  ESMP.ESMP_STAGGERLOC_CENTER_VCENTER)

        # Assign the cartesian coordinates of the grid centers in the ESMP_Grid
        # numpy.tile([a,b,c], 2) = [a,b,c,a,b,c]
        # numpy.repeat([a,b,c], 2) = [a,a,b,b,c,c]
        lons = numpy.deg2rad(center_lons_array)
        cos_lons = numpy.tile(numpy.cos(lons),
                              center_lats_array.shape[0] * center_levs_array.shape[0])
        sin_lons = numpy.tile(numpy.sin(lons),
                              center_lats_array.shape[0] * center_levs_array.shape[0])
        lats = numpy.deg2rad(center_lats_array)
        cos_lats = numpy.tile(numpy.repeat(numpy.cos(lats),
                                           center_lons_array.shape[0]),
                              center_levs_array.shape[0])
        sin_lats = numpy.tile(numpy.repeat(numpy.sin(lats),
                                           center_lons_array.shape[0]),
                              center_levs_array.shape[0])
        levs = center_levs_array / 1000.0
        if levels_are_depths:
            levs *= -1.0
        levs += self.__earth_rad
        levs /= self.__earth_rad
        levs = numpy.repeat(levs, center_lons_array.shape[0] * center_lats_array.shape[0])
        # XY plane through the prime meridian, Z toward central atlantic
        # 0 lon, 0 lat = (0, 0, R); 90 lon, 0 lat = (R, 0, 0); * lon, 90 lat = (0, R, 0)
        grid_x_coords[:] = levs * sin_lons * cos_lats
        grid_y_coords[:] = levs * sin_lats
        grid_z_coords[:] = levs * cos_lons * cos_lats

        if self.__debug:
            try:
                fout = open("rect_center_xyz.txt", "w")
                print >>fout, "rect_center_x = %s" % \
                    self.__myArrayStr(grid_x_coords, self.__rect_shape)
                print >>fout, "rect_center_y = %s" % \
                    self.__myArrayStr(grid_y_coords, self.__rect_shape)
                print >>fout, "rect_center_z = %s" % \
                    self.__myArrayStr(grid_z_coords, self.__rect_shape)
            finally:
                fout.close()

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
            if data_array.shape != self.__rect_shape:
                raise ValueError("data must have the shape " \
                                 "( len(center_lats), len(center_lats), len(center_levs) ) ")

            # Create the rectilinear source ESMP_Field if it does not exist
            if self.__rect_src_field == None:
                self.__rect_src_field = ESMP.ESMP_FieldCreateGrid(self.__rect_grid,
                                                       "rect_src_field",
                                                       ESMP.ESMP_TYPEKIND_R8,
                                                       ESMP.ESMP_STAGGERLOC_CENTER_VCENTER)

            # Retrieve the field data array in the ESMP_Field
            field_ptr = ESMP.ESMP_FieldGetPtr(self.__rect_src_field)

            # Assign the data to the field array in the ESMP_Field;
            # flatten in column-major (F) order to match lon-lat-lev assignment order
            field_ptr[:] = data_array.flatten('F')
            if self.__debug:
                fout = open("rect_data.txt", "w")
                try:
                    print >>fout, "rect_data = %s" % \
                        self.__myArrayStr(field_ptr, self.__rect_shape)
                finally:
                    fout.close()


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
            data: a 3D numpy array of data values located at the
                  rectilinear grid centers representing the regridded
                  curvilinear ESMP_Field data.  The undefined data
                  value will be assigned to unassigned data points.
        Raises:
            ValueError: if either the curvilinear source ESMP_Field
                        or the rectilinear destination ESMP_Field
                        does not exist.
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
        if self.__debug:
            fout = open("regr_rect_data.txt", "w")
            try:
                print >>fout, "regr_rect_data = %s" % \
                    self.__myArrayStr(result, self.__rect_shape)
            finally:
                fout.close()

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
            ValueError: if either the rectilinear source ESMP_Field
                        or the curvilinear destination ESMP_Field
                        does not exist.
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
        if self.__debug:
            fout = open("regr_curv_data.txt", "w")
            try:
                print >>fout, "regr_curv_data = %s" % \
                    self.__myArrayStr(result, self.__curv_shape)
            finally:
                fout.close()
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


    def __myArrayStr(self, arr, shape):
        '''
        Private utility method for debug printing of data arrays

        Arguments:
            arr -   numpy array (3D array flattened in Fortran order)
                    to be printed
            shape - tuple giving the 3D shape for the arr
        Returns:
            string representation of arr as
                "numpy.array([...]).reshape((...), order='F')"
            with newlines providing a block structure of the data.
            Data values are printed in format %#10.6f
        '''
        mystr = 'numpy.array([\n'
        k = 0
        for val in arr:
            mystr += '%#10.6f,' % val
            k += 1
            if k % shape[0] == 0:
                mystr += '\n'
            if k % (shape[0] * shape[1]) == 0:
                mystr += '\n'
        if mystr[-2:] == '\n\n':
            mystr = mystr[:-1] + ']'
        elif mystr[-1] == '\n':
            mystr += ']'
        else:
            mystr += '\n]'
        mystr += ").reshape(%s, order='F')" % str(shape)
        return mystr

