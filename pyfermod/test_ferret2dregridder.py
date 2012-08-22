#! python
#

'''
Unit tests for Ferret2DRegridder

@author: Karl Smith
'''
import unittest
import numpy
import ESMP
from ferret2dregridder import Ferret2DRegridder


class Ferret2DRegridderTests(unittest.TestCase):
    '''Unit tests for the Ferret2DRegridder class'''

    def setUp(self):
        # Use tuples for the arrays to make sure the NumPy 
        # arrays created in the class methods are always used;
        # not arrays that happened to be passed as input.
        # Also verifies passed data is not modified.
        self.curv_center_lons = (    (-14,  -6,   4),
                                     (-10,  -2,   6)    )
        self.curv_corner_lons = ( (-21, -11,  -5,   5),
                                  (-17,  -7,  -1,   9), 
                                  (-13,  -3,   3,  13)  )
        self.curv_center_lats = (     (50,  54,  58),
                                      (60,  64,  68)    )
        self.curv_corner_lats = (  (43,  47,  51,  55),
                                   (53,  57,  61,  65),
                                   (63,  67,  71,  75)  )
        # Ignore lon = 6, lat = 68
        self.curv_ignore_centers = ( (0, 0, 0),
                                     (0, 0, 1) )
        # Ignore lon = 13, lat = 75
        self.curv_ignore_corners = ( (0, 0, 0, 0),
                                     (0, 0, 0, 0),
                                     (0, 0, 0, 1) )

        data_array =  numpy.array(self.curv_center_lons) * \
                     (numpy.array(self.curv_center_lats) - 60.0)
        self.curv_data = tuple([ tuple(sublist) for sublist in data_array ])
         

        self.rect_edge_lons = (-23, -11, 1, 13)
        self.rect_edge_lats = (42, 51, 59, 68, 77)
        # Ignore lon = 7, lat = 72.5
        self.rect_ignore_centers = ( (0, 0, 0, 0),
                                     (0, 0, 0, 0),
                                     (0, 0, 0, 1) )
        # Ignore lon = 13, lat = 77
        self.rect_ignore_corners = ( (0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 1) )

        lon_edges = numpy.array(self.rect_edge_lons)
        lat_edges = numpy.array(self.rect_edge_lats)
        lon_ctr_vals = 0.5 * (lon_edges[:-1] + lon_edges[1:])
        lat_ctr_vals = 0.5 * (lat_edges[:-1] + lat_edges[1:])
        center_lats, center_lons = numpy.meshgrid(lat_ctr_vals, lon_ctr_vals)
        data_array =  center_lons * (center_lats - 60.0)
        self.rect_data = tuple([ tuple(sublist) for sublist in data_array ])
        ESMP.ESMP_Initialize()


    def testCreateCurvGrid(self):
        regridder = Ferret2DRegridder()

        # Test with all data given
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats, 
                                 self.curv_ignore_centers, self.curv_corner_lons, 
                                 self.curv_corner_lats, self.curv_ignore_corners)
        # Test overwriting with no flags 
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats, 
                                 None, self.curv_corner_lons, self.curv_corner_lats)

        # Test without corners
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats, 
                                 self.curv_ignore_centers)
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats)

        # Test invalid cases

        # Done with this regridder
        regridder.finalize()


    def testAssignCurvField(self):
        regridder = Ferret2DRegridder()

        # Test with completely specified grid
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats, 
                                 self.curv_ignore_centers, self.curv_corner_lons, 
                                 self.curv_corner_lats, self.curv_ignore_corners)
        regridder.assignCurvField()
        regridder.assignCurvField(self.curv_data)

        # Test overwriting with no flags 
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats, 
                                 None, self.curv_corner_lons, self.curv_corner_lats)
        regridder.assignCurvField(self.curv_data)
        regridder.assignCurvField()

        # Test without corners
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats, 
                                 self.curv_ignore_centers)
        regridder.assignCurvField(self.curv_data)
        regridder.assignCurvField()

        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats)
        regridder.assignCurvField()
        regridder.assignCurvField(self.curv_data)

        # Test invalid cases

        # Done with this regridder
        regridder.finalize()


    def testCreateRectGrid(self):
        regridder = Ferret2DRegridder()

        # Test with all data given
        regridder.createRectGrid(self.rect_edge_lons, self.rect_edge_lats, 
                                 self.rect_ignore_centers, self.rect_ignore_corners)

        # Test overwriting without center flags
        regridder.createRectGrid(self.rect_edge_lons, self.rect_edge_lats, 
                                 None, self.rect_ignore_corners)

        # Test without corners flags
        regridder.createRectGrid(self.rect_edge_lons, self.rect_edge_lats, 
                                 self.rect_ignore_centers)

        # Test with no flags 
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats) 

        # Test invalid cases

        # Done with this regridder
        regridder.finalize()


    def testAssignRectField(self):
        regridder = Ferret2DRegridder()

        # Test with fully specified grid
        regridder.createRectGrid(self.rect_edge_lons, self.rect_edge_lats, 
                                 self.rect_ignore_centers, self.rect_ignore_corners)
        regridder.assignRectField(self.curv_data)
        regridder.assignRectField()

        # Test overwriting without center flags
        regridder.createRectGrid(self.rect_edge_lons, self.rect_edge_lats, 
                                 None, self.rect_ignore_corners)
        regridder.assignRectField()
        regridder.assignRectField(self.curv_data)

        # Test without corners flags
        regridder.createRectGrid(self.rect_edge_lons, self.rect_edge_lats, 
                                 self.rect_ignore_centers)
        regridder.assignRectField()
        regridder.assignRectField(self.curv_data)

        # Test with no flags 
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats) 
        regridder.assignRectField(self.curv_data)
        regridder.assignRectField()

        # Test invalid cases

        # Done with this regridder
        regridder.finalize()


    def testRegridCurvToRect(self):
        self.fail("Not implemented")


    def testRegridRectToCurv(self):
        self.fail("Not implemented")


    def tearDown(self):
        ESMP.ESMP_Finalize()


if __name__ == "__main__":
    unittest.main()

