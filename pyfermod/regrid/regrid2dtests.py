'''
Unit tests for CurvRectRegridder

@author: Karl Smith
'''

from __future__ import print_function

import unittest
import numpy
import ESMP
from esmpcontrol import ESMPControl
from regrid2d import CurvRectRegridder


class CurvRectRegridderTests(unittest.TestCase):
    '''
    Unit tests for the CurvRectRegridder class
    '''

    # flag to indicate when to call ESMPControl().stopESMP()
    last_test = False


    def setUp(self):
        '''
        Create some repeatedly used test data.
        '''
        # Use tuples for the arrays to make sure the NumPy 
        # arrays created in the class methods are always used;
        # not arrays that happened to be passed as input.
        # Also verifies passed data is not modified.

        # Rectilinear coordinates, data. and flags
        crn_lons = numpy.linspace(-110, -90, 11)
        crn_lats = numpy.linspace(0, 32, 9)
        ctr_lons = 0.5 * (crn_lons[:-1] + crn_lons[1:])
        ctr_lats = 0.5 * (crn_lats[:-1] + crn_lats[1:])
        ctr_lats_mat, ctr_lons_mat = numpy.meshgrid(ctr_lats, ctr_lons)
        data = -2.0 * numpy.sin(numpy.deg2rad(ctr_lons_mat)) \
                    * numpy.cos(numpy.deg2rad(ctr_lats_mat))
        ctr_flags = numpy.zeros(data.shape, dtype=numpy.int32)
        ctr_flags[:2, :2] = 1
        crn_flags = numpy.zeros((crn_lons.shape[0], crn_lats.shape[0]), dtype=numpy.int32)
        crn_flags[:2, :2] = 1

        # Turn rectilinear arrays into tuples
        self.rect_corner_lons = tuple(crn_lons)
        self.rect_corner_lats = tuple(crn_lats)
        self.rect_center_lons = tuple(ctr_lons)
        self.rect_center_lats = tuple(ctr_lats)
        self.rect_center_ignr = tuple([tuple(subarr) for subarr in ctr_flags.tolist()])
        self.rect_corner_ignr = tuple([tuple(subarr) for subarr in crn_flags.tolist()])
        self.rect_data = tuple([tuple(subarr) for subarr in data.tolist()])

        # Curvilinear coordindates - one step further out on all sides of the region
        crn_lons = numpy.linspace(-112, -88, 13)
        crn_lats = numpy.linspace(-4, 36, 11)
        crn_lats_mat, crn_lons_mat = numpy.meshgrid(crn_lats, crn_lons)
        ctr_lons = 0.5 * (crn_lons[:-1] + crn_lons[1:])
        ctr_lats = 0.5 * (crn_lats[:-1] + crn_lats[1:])
        ctr_lats_mat, ctr_lons_mat = numpy.meshgrid(ctr_lats, ctr_lons)
        # Pull coordinates in some towards the center
        crn_lons = crn_lons_mat * numpy.cos(numpy.deg2rad(crn_lats_mat - 16.0) / 2.0)
        crn_lats = crn_lats_mat * numpy.cos(numpy.deg2rad(crn_lons_mat + 100.0) / 2.0)
        ctr_lons = ctr_lons_mat * numpy.cos(numpy.deg2rad(ctr_lats_mat - 16.0) / 2.0)
        ctr_lats = ctr_lats_mat * numpy.cos(numpy.deg2rad(ctr_lons_mat + 100.0) / 2.0)
        # Curvilinear data and flags
        data = -2.0 * numpy.sin(numpy.deg2rad(ctr_lons)) \
                    * numpy.cos(numpy.deg2rad(ctr_lats))
        ctr_flags = numpy.zeros(data.shape, dtype=numpy.int32)
        ctr_flags[:3, :3] = 1
        crn_flags = numpy.zeros(crn_lons.shape, dtype=numpy.int32)
        crn_flags[:3, :3] = 1

        # Turn curvilinear arrays into tuples
        self.curv_corner_lons = tuple([tuple(subarr) for subarr in crn_lons.tolist()])
        self.curv_corner_lats = tuple([tuple(subarr) for subarr in crn_lats.tolist()])
        self.curv_center_lons = tuple([tuple(subarr) for subarr in ctr_lons.tolist()])
        self.curv_center_lats = tuple([tuple(subarr) for subarr in ctr_lats.tolist()])
        self.curv_center_ignr = tuple([tuple(subarr) for subarr in ctr_flags.tolist()])
        self.curv_corner_ignr = tuple([tuple(subarr) for subarr in crn_flags.tolist()])
        self.curv_data = tuple([tuple(subarr) for subarr in data.tolist()])

        # undef_val must be a numpy array
        self.undef_val = numpy.array([1.0E10], dtype=numpy.float64)

        if not ESMPControl().startCheckESMP():
            self.fail("startCheckESMP did not succeed - test called after last_test set to True")


    def test01CurvRectRegridderInit(self):
        '''
        Test of the CurvRectRegridder.__init__ method.
        '''
        regridder = CurvRectRegridder()
        self.assertTrue(regridder != None, "CurvRectRegridder() returned None")
        regridder.finalize()


    def test02CreateCurvGrid(self):
        '''
        Tests the CurvRectRegridder.createCurvGrid method.
        Since nothing is returned from this method, just
        checks for unexpected/expected Errors being raised.
        '''
        regridder = CurvRectRegridder()

        # Test with all corner and center data
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_ignr, self.curv_corner_lons,
                                 self.curv_corner_lats, self.curv_corner_ignr)

        # Test without flags 
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 None, self.curv_corner_lons, self.curv_corner_lats)

        # Test without corners
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_ignr)

        # Test without corners or flags
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats)

        # TODO: Test invalid cases

        # Done with this regridder
        regridder.finalize()


    def test03AssignCurvField(self):
        '''
        Tests the CurvRectRegridder.assignCurvGrid method.
        Since nothing is returned from this method, just
        checks for unexpected/expected Errors being raised.
        '''
        regridder = CurvRectRegridder()

        # Test with all corner and center data
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_ignr, self.curv_corner_lons,
                                 self.curv_corner_lats, self.curv_corner_ignr)
        regridder.assignCurvField()
        regridder.assignCurvField(self.curv_data)

        # Test without flags 
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 None, self.curv_corner_lons, self.curv_corner_lats)
        regridder.assignCurvField(self.curv_data)
        regridder.assignCurvField()

        # Test without corners
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_ignr)
        regridder.assignCurvField(self.curv_data)
        regridder.assignCurvField()

        # Test without corners or flags
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats)
        regridder.assignCurvField()
        regridder.assignCurvField(self.curv_data)

        # TODO: Test invalid cases

        # Done with this regridder
        regridder.finalize()


    def test04CreateRectGrid(self):
        '''
        Tests the CurvRectRegridder.createRectGrid method.
        Since nothing is returned from this method, just
        checks for unexpected/expected Errors being raised.
        '''
        regridder = CurvRectRegridder()

        # Test with all corner and center data
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_ignr, self.rect_corner_lons,
                                 self.rect_corner_lats, self.rect_corner_ignr)

        # Test without flags
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 None, self.rect_corner_lons, self.rect_corner_lats)

        # Test without corners
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_ignr)

        # Test without corners or flags 
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats)

        # TODO: Test invalid cases

        # Done with this regridder
        regridder.finalize()


    def test05AssignRectField(self):
        '''
        Tests the CurvRectRegridder.assignRectGrid method.
        Since nothing is returned from this method, just
        checks for unexpected/expected Errors being raised.
        '''
        regridder = CurvRectRegridder()

        # Test with all corner and center data
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_ignr, self.rect_corner_lons,
                                 self.rect_corner_lats, self.rect_corner_ignr)
        regridder.assignRectField(self.rect_data)
        regridder.assignRectField()

        # Test without flags
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 None, self.rect_corner_lons, self.rect_corner_lats)
        regridder.assignRectField()
        regridder.assignRectField(self.rect_data)

        # Test without corners
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_ignr)
        regridder.assignRectField()
        regridder.assignRectField(self.rect_data)

        # Test without corners or flags 
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats)
        regridder.assignRectField(self.rect_data)
        regridder.assignRectField()

        # TODO: Test invalid cases

        # Done with this regridder
        regridder.finalize()


    def test06RegridCurvToRectConserve(self):
        '''
        Tests the CurvRectRegridder.regridCurvToRect method using conservative regridding
        '''
        regridder = CurvRectRegridder()

        # Test with all corner and center data, using conservative regridding
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_ignr, self.curv_corner_lons,
                                 self.curv_corner_lats, self.curv_corner_ignr)
        regridder.assignCurvField(self.curv_data)
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_ignr, self.rect_corner_lons,
                                 self.rect_corner_lats, self.rect_corner_ignr)
        regridder.assignRectField()
        regrid_data = regridder.regridCurvToRect(self.undef_val, 
                                                 ESMP.ESMP_REGRIDMETHOD_CONSERVE)
        expect_data = numpy.array(self.rect_data, dtype=numpy.float64)
        undef_flags = numpy.array(self.rect_center_ignr, dtype=numpy.bool)
        expect_data[undef_flags] = self.undef_val
        mismatch_found = False
        # Couple "good" points next to ignored data area are a bit wonky
        expect_data[2, 0] = self.undef_val
        regrid_data[2, 0] = self.undef_val
        expect_data[2, 1] = self.undef_val
        regrid_data[2, 1] = self.undef_val
        for i in range(expect_data.shape[0]):
            for j in range(expect_data.shape[1]):
                if numpy.abs(expect_data[i, j] - regrid_data[i, j]) > 0.0007:
                    mismatch_found = True
                    print("expect = %#6.4f, found = %#6.4f for lon = %5.1f, " \
                          "lat = %5.1f" % (expect_data[i, j], regrid_data[i, j],
                           self.rect_center_lons[i], self.rect_center_lats[j]))
        if mismatch_found:
            self.fail("data mismatch found")


    def test07RegridCurvToRectBilinear(self):
        '''
        Tests the CurvRectRegridder.regridCurvToRect method using bilinear regridding
        '''
        regridder = CurvRectRegridder()

        # Test with only center data and no flags, using bilinear regridding
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats)
        regridder.assignCurvField(self.curv_data)
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats)
        regridder.assignRectField()
        regrid_data = regridder.regridCurvToRect(self.undef_val,
                                                 ESMP.ESMP_REGRIDMETHOD_BILINEAR)
        expect_data = numpy.array(self.rect_data, dtype=numpy.float64)
        mismatch_found = False
        # one point falls outside the curvilinear centerpoints grid?
        expect_data[5, 0] = self.undef_val
        for i in range(expect_data.shape[0]):
            for j in range(expect_data.shape[1]):
                if numpy.abs(expect_data[i, j] - regrid_data[i, j]) > 0.0003:
                    mismatch_found = True
                    print("expect = %#6.4f, found = %#6.4f for lon = %5.1f, " \
                          "lat = %5.1f" % (expect_data[i, j], regrid_data[i, j],
                           self.rect_center_lons[i], self.rect_center_lats[j]))
        if mismatch_found:
            self.fail("data mismatch found")


    def test08RegridCurvToRectPatch(self):
        '''
        Tests the CurvRectRegridder.regridCurvToRect method using patch regridding
        '''
        regridder = CurvRectRegridder()

        # Test with only center data, and flags only on rectilinear centers,
        # using patch regridding
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats)
        regridder.assignCurvField(self.curv_data)
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_ignr)
        regridder.assignRectField()
        regrid_data = regridder.regridCurvToRect(self.undef_val,
                                                 ESMP.ESMP_REGRIDMETHOD_PATCH)
        expect_data = numpy.array(self.rect_data, dtype=numpy.float64)
        undef_flags = numpy.array(self.rect_center_ignr, dtype=numpy.bool)
        expect_data[undef_flags] = self.undef_val
        # one point falls outside the curvilinear centerpoints grid?
        expect_data[5, 0] = self.undef_val
        mismatch_found = False
        for i in range(expect_data.shape[0]):
            for j in range(expect_data.shape[1]):
                if numpy.abs(expect_data[i, j] - regrid_data[i, j]) > 0.0011:
                    mismatch_found = True
                    print("expect = %#6.4f, found = %#6.4f for lon = %5.1f, " \
                          "lat = %5.1f" % (expect_data[i, j], regrid_data[i, j],
                           self.rect_center_lons[i], self.rect_center_lats[j]))
        if mismatch_found:
            self.fail("data mismatch found")


    def test09RegridRectToCurvConserve(self):
        '''
        Tests the CurvRectRegridder.regridRectToCurv method using conservative regridding
        '''
        regridder = CurvRectRegridder()

        # Test with all corner and center data, using conservative regridding
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_ignr, self.curv_corner_lons,
                                 self.curv_corner_lats, self.curv_corner_ignr)
        regridder.assignCurvField()
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_ignr, self.rect_corner_lons,
                                 self.rect_corner_lats, self.rect_corner_ignr)
        regridder.assignRectField(self.rect_data)
        regrid_data = regridder.regridRectToCurv(self.undef_val,
                                                 ESMP.ESMP_REGRIDMETHOD_CONSERVE)
        expect_data = numpy.array(self.curv_data, dtype=numpy.float64)
        undef_flags = numpy.array(self.curv_center_ignr, dtype=numpy.bool)
        expect_data[undef_flags] = self.undef_val
        # Couple "good" points next to ignored area are a bit wonky
        expect_data[1, 3] = self.undef_val
        regrid_data[1, 3] = self.undef_val
        expect_data[2, 3] = self.undef_val
        regrid_data[2, 3] = self.undef_val
        mismatch_found = False
        # Ignore outermost edges of curvilinear grid since
        # they aren't really well covered by the rectilinear grid
        # Also ignore the second east-most edge;
        # also not well covered and errors are larger 
        for i in range(1, expect_data.shape[0] - 2):
            for j in range(1, expect_data.shape[1] - 1):
                if numpy.abs(expect_data[i, j] - regrid_data[i, j]) > 0.0004:
                    mismatch_found = True
                    print("expect = %#6.4f, found = %#6.4f for lon = %7.3f, " \
                          "lat = %7.3f" % (expect_data[i, j], regrid_data[i, j], 
                          self.curv_center_lons[i][j], self.curv_center_lats[i][j]))
        if mismatch_found:
            self.fail("data mismatch found")


    def test10RegridRectToCurvBilinear(self):
        '''
        Tests the CurvRectRegridder.regridRectToCurv method using bilinear regridding
        '''
        regridder = CurvRectRegridder()

        # Test with only center data and no flags, using bilinear regridding
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats)
        regridder.assignCurvField()
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats)
        regridder.assignRectField(self.rect_data)
        regrid_data = regridder.regridRectToCurv(self.undef_val,
                                                 ESMP.ESMP_REGRIDMETHOD_BILINEAR)
        expect_data = numpy.array(self.curv_data, dtype=numpy.float64)
        mismatch_found = False
        # Ignore outermost edges of curvilinear grid since
        # they aren't really well covered by the rectilinear grid
        # Also ignore the second east-most edge and second south-most edge;
        # also not covered
        for i in range(1, expect_data.shape[0] - 2):
            for j in range(2, expect_data.shape[1] - 1):
                if numpy.abs(expect_data[i, j] - regrid_data[i, j]) > 0.0003:
                    mismatch_found = True
                    print("expect = %#6.4f, found = %#6.4f for lon = %7.3f, " \
                          "lat = %7.3f" % (expect_data[i, j], regrid_data[i, j],
                          self.curv_center_lons[i][j], self.curv_center_lats[i][j]))
        if mismatch_found:
            self.fail("data mismatch found")


    def test11RegridRectToCurvPatch(self):
        '''
        Tests the CurvRectRegridder.regridRectToCurv method using patch regridding
        '''
        # Mark as the last test so ESMPControl().stopESMP will be called
        self.last_test = True

        regridder = CurvRectRegridder()

        # Test with only center data, and flags only on curvilinear centers,
        # using patch regridding
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_ignr)
        regridder.assignCurvField()
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats)
        regridder.assignRectField(self.rect_data)
        regrid_data = regridder.regridRectToCurv(self.undef_val,
                                                 ESMP.ESMP_REGRIDMETHOD_PATCH)
        expect_data = numpy.array(self.curv_data, dtype=numpy.float64)
        undef_flags = numpy.array(self.curv_center_ignr, dtype=numpy.bool)
        expect_data[undef_flags] = self.undef_val
        mismatch_found = False
        # Ignore outermost edges of curvilinear grid since
        # they aren't really well covered by the rectilinear grid
        # Also ignore the second east-most edge and second south-most edge;
        # also not covered
        for i in range(1, expect_data.shape[0] - 2):
            for j in range(2, expect_data.shape[1] - 1):
                if numpy.abs(expect_data[i, j] - regrid_data[i, j]) > 0.0011:
                    mismatch_found = True
                    print("expect = %#6.4f, found = %#6.4f for lon = %7.3f, " \
                          "lat = %7.3f" % (expect_data[i, j], regrid_data[i, j],
                          self.curv_center_lons[i][j], self.curv_center_lats[i][j]))
        if mismatch_found:
            self.fail("data mismatch found")


    def tearDown(self):
        '''
        Finalize ESMP if it has been initialized and if this is the last test
        '''
        if self.last_test:
            ESMPControl().stopESMP(True)


if __name__ == "__main__":
    '''
    Run the unit tests in this module.
    '''
    unittest.main()

