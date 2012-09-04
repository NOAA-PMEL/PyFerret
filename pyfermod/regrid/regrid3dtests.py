'''
Unit tests for CurvRect3DRegridder

@author: Karl Smith
'''
import unittest
import numpy
import ESMP
from esmpcontrol import ESMPControl
from regrid3d import CurvRect3DRegridder


class CurvRect3DRegridderTests(unittest.TestCase):
    '''
    Unit tests for the CurvRect3DRegridder class
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

        # Rectilinear coordinates and data
        crn_lons = numpy.linspace(-16.0,  16.0, 17)
        crn_lats = numpy.linspace(-11.0,  11.0, 12)
        crn_levs = numpy.linspace( 10.0, 100.0, 10)
        crn_shape = (crn_lons.shape[0], crn_lats.shape[0], crn_levs.shape[0])

        ctr_lons = 0.5 * (crn_lons[:-1] + crn_lons[1:])
        ctr_lats = 0.5 * (crn_lats[:-1] + crn_lats[1:])
        ctr_levs = 0.5 * (crn_levs[:-1] + crn_levs[1:])
        ctr_shape = (ctr_lons.shape[0], ctr_lats.shape[0], ctr_levs.shape[0])

        # Create in C order so the tuplets are [lon][lat][lev] 
        ctr_lons_mat = numpy.repeat(ctr_lons, ctr_shape[1] * ctr_shape[2]) \
                            .reshape(ctr_shape)
        ctr_lats_mat = numpy.repeat(numpy.tile(ctr_lats, ctr_shape[0]),
                                    ctr_shape[2]) \
                            .reshape(ctr_shape)
        ctr_levs_mat = numpy.tile(ctr_levs, ctr_shape[0] * ctr_shape[1]) \
                            .reshape(ctr_shape)

        data = 2.0 * numpy.cos(numpy.deg2rad(ctr_lons_mat + 20.0)) \
                   * numpy.cos(numpy.deg2rad(ctr_lats_mat + 15.0)) \
                   / numpy.log10(ctr_levs_mat + 10.0)

        ctr_flags = numpy.zeros(ctr_shape, dtype=numpy.int32)
        ctr_flags[ 0,  0, :] = 1
        ctr_flags[ 0, -1, :] = 1
        ctr_flags[-1, -1, :] = 1

        crn_flags = numpy.zeros(crn_shape, dtype=numpy.int32)
        crn_flags[ 0,  0, :] = 1
        crn_flags[ 0, -1, :] = 1
        crn_flags[-1, -1, :] = 1

        self.rect_corner_lons = tuple(crn_lons)
        self.rect_corner_lats = tuple(crn_lats)
        self.rect_corner_levs = tuple(crn_levs)
        self.rect_center_lons = tuple(ctr_lons)
        self.rect_center_lats = tuple(ctr_lats)
        self.rect_center_levs = tuple(ctr_levs)
        self.rect_center_ignr = tuple([tuple([tuple(subsubarr) for subsubarr in subarr])
                                       for subarr in ctr_flags.tolist()])
        self.rect_corner_ignr = tuple([tuple([tuple(subsubarr) for subsubarr in subarr])
                                       for subarr in crn_flags.tolist()])
        self.rect_data        = tuple([tuple([tuple(subsubarr) for subsubarr in subarr])
                                       for subarr in data.tolist()])

        # Curvilinear coordinates and data - expanded out a step on all region sides
        crn_lons = numpy.linspace(-18.0,  18.0, 19)
        crn_lats = numpy.linspace(-13.0,  13.0, 14)
        crn_levs = numpy.linspace(  0.0, 110.0, 12)
        crn_shape = (crn_lons.shape[0], crn_lats.shape[0], crn_levs.shape[0])

        ctr_lons = 0.5 * (crn_lons[:-1] + crn_lons[1:])
        ctr_lats = 0.5 * (crn_lats[:-1] + crn_lats[1:])
        ctr_levs = 0.5 * (crn_levs[:-1] + crn_levs[1:])
        ctr_shape = (ctr_lons.shape[0], ctr_lats.shape[0], ctr_levs.shape[0])

        # Create in C order so the tuplets are [lon][lat][lev] 
        crn_lons_mat = numpy.repeat(crn_lons, crn_shape[1] * crn_shape[2]) \
                            .reshape(crn_shape)
        crn_lats_mat = numpy.repeat(numpy.tile(crn_lats, crn_shape[0]),
                                    crn_shape[2]) \
                            .reshape(crn_shape)
        crn_levs_mat = numpy.tile(crn_levs, crn_shape[0] * crn_shape[1]) \
                            .reshape(crn_shape)

        # Create in C order so the tuplets are [lon][lat][lev] 
        ctr_lons_mat = numpy.repeat(ctr_lons, ctr_shape[1] * ctr_shape[2]) \
                            .reshape(ctr_shape)
        ctr_lats_mat = numpy.repeat(numpy.tile(ctr_lats, ctr_shape[0]),
                                    ctr_shape[2]) \
                            .reshape(ctr_shape)
        ctr_levs_mat = numpy.tile(ctr_levs, ctr_shape[0] * ctr_shape[1]) \
                            .reshape(ctr_shape)

        # Tweak the coordinates some, pulling in near edges of the region
        crn_lons = crn_lons_mat * numpy.cos(numpy.deg2rad(crn_lats_mat))
        crn_lats = crn_lats_mat * numpy.cos(numpy.deg2rad(crn_lons_mat))
        crn_levs = crn_levs_mat * numpy.cos(numpy.deg2rad(crn_lats_mat) / 2.0) \
                                * numpy.cos(numpy.deg2rad(crn_lons_mat) / 2.0) \
                                * numpy.cos(numpy.deg2rad(crn_levs_mat - 55.0) / 5.0)

        ctr_lons = ctr_lons_mat * numpy.cos(numpy.deg2rad(ctr_lats_mat))
        ctr_lats = ctr_lats_mat * numpy.cos(numpy.deg2rad(ctr_lons_mat))
        ctr_levs = ctr_levs_mat * numpy.cos(numpy.deg2rad(ctr_lats_mat) / 2.0) \
                                * numpy.cos(numpy.deg2rad(ctr_lons_mat) / 2.0) \
                                * numpy.cos(numpy.deg2rad(ctr_levs_mat - 55.0) / 5.0)

        data = 2.0 * numpy.cos(numpy.deg2rad(ctr_lons + 20.0)) \
                   * numpy.cos(numpy.deg2rad(ctr_lats + 15.0)) \
                   / numpy.log10(ctr_levs + 10.0)

        ctr_flags = numpy.zeros(ctr_shape, dtype=numpy.int32)
        ctr_flags[ :2,  :2, :] = 1
        ctr_flags[ :2, -2:, :] = 1
        ctr_flags[-2:, -2:, :] = 1

        crn_flags = numpy.zeros(crn_shape, dtype=numpy.int32)
        crn_flags[ :2,  :2, :] = 1
        crn_flags[ :2, -2:, :] = 1
        crn_flags[-2:, -2:, :] = 1
 
        self.curv_corner_lons = tuple([tuple([tuple(subsubarr) for subsubarr in subarr])
                                       for subarr in crn_lons.tolist()])
        self.curv_corner_lats = tuple([tuple([tuple(subsubarr) for subsubarr in subarr])
                                       for subarr in crn_lats.tolist()])
        self.curv_corner_levs = tuple([tuple([tuple(subsubarr) for subsubarr in subarr])
                                       for subarr in crn_levs.tolist()])
        self.curv_center_lons = tuple([tuple([tuple(subsubarr) for subsubarr in subarr])
                                       for subarr in ctr_lons.tolist()])
        self.curv_center_lats = tuple([tuple([tuple(subsubarr) for subsubarr in subarr])
                                       for subarr in ctr_lats.tolist()])
        self.curv_center_levs = tuple([tuple([tuple(subsubarr) for subsubarr in subarr])
                                       for subarr in ctr_levs.tolist()])
        self.curv_center_ignr = tuple([tuple([tuple(subsubarr) for subsubarr in subarr])
                                       for subarr in ctr_flags.tolist()])
        self.curv_corner_ignr = tuple([tuple([tuple(subsubarr) for subsubarr in subarr])
                                       for subarr in crn_flags.tolist()])
        self.curv_data        = tuple([tuple([tuple(subsubarr) for subsubarr in subarr])
                                       for subarr in data.tolist()])

        # undef_val must be a numpy array
        self.undef_val = numpy.array([1.0E10], dtype=numpy.float64)

        if not ESMPControl().startCheckESMP():
            self.fail("startCheckESMP did not succeed - test called after last_test set to True")


    def test01CurvRectRegridderInit(self):
        '''
        Test of the CurvRect3DRegridder.__init__ method.
        '''
        regridder = CurvRect3DRegridder()
        self.assertTrue(regridder != None, "CurvRect3DRegridder() returned None")
        regridder.finalize()


    def test02CreateCurvGrid(self):
        '''
        Tests the CurvRect3DRegridder.createCurvGrid method.
        Since nothing is returned from this method, just
        checks for unexpected/expected Errors being raised.
        '''
        regridder = CurvRect3DRegridder()

        # Test with all corner and center data
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_levs, self.curv_center_ignr,
                                 True,
                                 self.curv_corner_lons, self.curv_corner_lats,
                                 self.curv_corner_levs, self.curv_corner_ignr)

        # Test without flags 
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_levs, None, True,
                                 self.curv_corner_lons, self.curv_corner_lats,
                                 self.curv_corner_levs)

        # Test without corners
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_levs, self.curv_center_ignr)

        # Test without corners or flags
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_levs)

        # Test invalid cases

        # Done with this regridder
        regridder.finalize()


    def test03AssignCurvField(self):
        '''
        Tests the CurvRect3DRegridder.assignCurvGrid method.
        Since nothing is returned from this method, just
        checks for unexpected/expected Errors being raised.
        '''
        regridder = CurvRect3DRegridder()

        # Test with all corner and center data
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_levs, self.curv_center_ignr,
                                 True,
                                 self.curv_corner_lons, self.curv_corner_lats,
                                 self.curv_corner_levs, self.curv_corner_ignr)
        regridder.assignCurvField()
        regridder.assignCurvField(self.curv_data)

        # Test without flags 
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_levs, None, True,
                                 self.curv_corner_lons, self.curv_corner_lats,
                                 self.curv_corner_levs)
        regridder.assignCurvField(self.curv_data)
        regridder.assignCurvField()

        # Test without corners
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_levs, self.curv_center_ignr)
        regridder.assignCurvField(self.curv_data)
        regridder.assignCurvField()

        # Test without corners or flags
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_levs)
        regridder.assignCurvField()
        regridder.assignCurvField(self.curv_data)

        # Test invalid cases

        # Done with this regridder
        regridder.finalize()


    def test04CreateRectGrid(self):
        '''
        Tests the CurvRect3DRegridder.createRectGrid method.
        Since nothing is returned from this method, just
        checks for unexpected/expected Errors being raised.
        '''
        regridder = CurvRect3DRegridder()

        # Test with all corner and center data
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_levs, self.rect_center_ignr,
                                 True,
                                 self.rect_corner_lons, self.rect_corner_lats,
                                 self.rect_corner_levs, self.rect_corner_ignr)

        # Test without flags
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_levs, None, True,
                                 self.rect_corner_lons, self.rect_corner_lats,
                                 self.rect_corner_levs)

        # Test without corners
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_levs, self.rect_center_ignr)

        # Test without corners or flags 
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_levs)

        # Test invalid cases

        # Done with this regridder
        regridder.finalize()


    def test05AssignRectField(self):
        '''
        Tests the CurvRect3DRegridder.assignRectGrid method.
        Since nothing is returned from this method, just
        checks for unexpected/expected Errors being raised.
        '''
        regridder = CurvRect3DRegridder()

        # Test with all corner and center data
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_levs, self.rect_center_ignr,
                                 True,
                                 self.rect_corner_lons, self.rect_corner_lats,
                                 self.rect_corner_levs, self.rect_corner_ignr)
        regridder.assignRectField(self.rect_data)
        regridder.assignRectField()

        # Test without flags
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_levs, None, True,
                                 self.rect_corner_lons, self.rect_corner_lats,
                                 self.rect_corner_levs)
        regridder.assignRectField()
        regridder.assignRectField(self.rect_data)

        # Test without corners
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_levs, self.rect_center_ignr)
        regridder.assignRectField()
        regridder.assignRectField(self.rect_data)

        # Test without corners or flags 
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_levs)
        regridder.assignRectField(self.rect_data)
        regridder.assignRectField()

        # Test invalid cases

        # Done with this regridder
        regridder.finalize()


    def test06RegridCurvToRectBilinear(self):
        '''
        Tests the CurvRect3DRegridder.regridCurvToRect method.
        '''
        margin = 1
        delta = 1.0
      
        regridder = CurvRect3DRegridder()

        # Test with only center data but no flags
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_levs)
        regridder.assignCurvField(self.curv_data)
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_levs)
        regridder.assignRectField()
        regrid_data = regridder.regridCurvToRect(self.undef_val,
                                                 ESMP.ESMP_REGRIDMETHOD_BILINEAR)
        expect_data = numpy.array(self.rect_data, dtype=numpy.float64)
        mismatch_found = False
        for i in xrange(margin, expect_data.shape[0] - margin):
            for j in xrange(margin, expect_data.shape[1] - margin):
                for k in xrange(margin, expect_data.shape[2] - margin):
                    if numpy.abs(expect_data[i, j, k] - regrid_data[i, j, k]) > delta:
                        mismatch_found = True
                        print "expect = %#6.4f, found = %#6.4f for lon = %5.1f, " \
                              "lat = %5.1f, lev = %5.1f" % (expect_data[i, j, k], 
                              regrid_data[i, j, k], self.rect_center_lons[i], 
                              self.rect_center_lats[j], self.rect_center_levs[k])
        if mismatch_found:
            self.fail("data mismatch found for bilinear regridding without flags")

        # Test with only center data and with flags
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_levs, self.curv_center_ignr)
        regridder.assignCurvField(self.curv_data)
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_levs, self.rect_center_ignr)
        regridder.assignRectField()
        regrid_data = regridder.regridCurvToRect(self.undef_val,
                                                 ESMP.ESMP_REGRIDMETHOD_BILINEAR)
        expect_data = numpy.array(self.rect_data, dtype=numpy.float64)
        undef_flags = numpy.array(self.rect_center_ignr, dtype=numpy.bool)
        expect_data[undef_flags] = self.undef_val
        mismatch_found = False
        for i in xrange(margin, expect_data.shape[0] - margin):
            for j in xrange(margin, expect_data.shape[1] - margin):
                for k in xrange(margin, expect_data.shape[2] - margin):
                    if numpy.abs(expect_data[i, j, k] - regrid_data[i, j, k]) > delta:
                        mismatch_found = True
                        print "expect = %#6.4f, found = %#6.4f for lon = %5.1f, " \
                              "lat = %5.1f, lev = %5.1f" % (expect_data[i, j, k], 
                              regrid_data[i, j, k], self.rect_center_lons[i], 
                              self.rect_center_lats[j], self.rect_center_levs[k])
        if mismatch_found:
            self.fail("data mismatch found for bilinear regridding with flags")


    def test07RegridCurvToRectConserve(self):
        '''
        Tests the CurvRect3DRegridder.regridCurvToRect method.
        '''
        margin = 1
        delta = 1.0
      
        regridder = CurvRect3DRegridder()

        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_levs, None, True,
                                 self.curv_corner_lons, self.curv_corner_lats,
                                 self.curv_corner_levs)
        regridder.assignCurvField(self.curv_data)
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_levs, None, True,
                                 self.rect_corner_lons, self.rect_corner_lats,
                                 self.rect_corner_levs)
        regridder.assignRectField()
        regrid_data = regridder.regridCurvToRect(self.undef_val, 
                                                 ESMP.ESMP_REGRIDMETHOD_CONSERVE)
        expect_data = numpy.array(self.rect_data, dtype=numpy.float64)
        mismatch_found = False
        for i in xrange(margin, expect_data.shape[0] - margin):
            for j in xrange(margin, expect_data.shape[1] - margin):
                for k in xrange(margin, expect_data.shape[2] - margin):
                    if numpy.abs(expect_data[i, j, k] - regrid_data[i, j, k]) > delta:
                        mismatch_found = True
                        print "expect = %#6.4f, found = %#6.4f for lon = %5.1f, " \
                              "lat = %5.1f, lev = %5.1f" % (expect_data[i, j, k], 
                              regrid_data[i, j, k], self.rect_center_lons[i], 
                              self.rect_center_lats[j], self.rect_center_levs[k])
        if mismatch_found:
            self.fail("data mismatch found for conservative regridding")



    def test08RegridCurvToRectPatch(self):
        '''
        Tests the CurvRect3DRegridder.regridCurvToRect method.
        '''
        margin = 1
        delta = 1.0
      
        regridder = CurvRect3DRegridder()

        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_levs)
        regridder.assignCurvField(self.curv_data)
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_levs)
        regridder.assignRectField()
        regrid_data = regridder.regridCurvToRect(self.undef_val,
                                                 ESMP.ESMP_REGRIDMETHOD_PATCH)
        expect_data = numpy.array(self.rect_data, dtype=numpy.float64)
        mismatch_found = False
        for i in xrange(margin, expect_data.shape[0] - margin):
            for j in xrange(margin, expect_data.shape[1] - margin):
                for k in xrange(margin, expect_data.shape[2] - margin):
                    if numpy.abs(expect_data[i, j, k] - regrid_data[i, j, k]) > delta:
                        mismatch_found = True
                        print "expect = %#6.4f, found = %#6.4f for lon = %5.1f, " \
                              "lat = %5.1f, lev = %5.1f" % (expect_data[i, j, k], 
                              regrid_data[i, j, k], self.rect_center_lons[i], 
                              self.rect_center_lats[j], self.rect_center_levs[k])
        if mismatch_found:
            self.fail("data mismatch found for patch regridding")


    def test09RegridRectToCurvBilinear(self):
        '''
        Tests the CurvRect3DRegridder.regridRectToCurv method.
        '''
        margin = 2
        delta = 1.0

        regridder = CurvRect3DRegridder()

        # Test with only center data but no flags
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_levs)
        regridder.assignCurvField()
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_levs)
        regridder.assignRectField(self.rect_data)
        regrid_data = regridder.regridRectToCurv(self.undef_val,
                                                 ESMP.ESMP_REGRIDMETHOD_BILINEAR)
        expect_data = numpy.array(self.curv_data, dtype=numpy.float64)
        mismatch_found = False
        for i in xrange(margin, expect_data.shape[0] - margin):
            for j in xrange(margin, expect_data.shape[1] - margin):
                for k in xrange(margin, expect_data.shape[2] - margin):
                    if numpy.abs(expect_data[i, j, k] - regrid_data[i, j, k]) > delta:
                        mismatch_found = True
                        print "expect = %#6.4f, found = %#6.4f for lon = %5.1f, " \
                              "lat = %5.1f, lev = %5.1f" % (expect_data[i, j, k], 
                              regrid_data[i, j, k], self.curv_center_lons[i][j][k], 
                              self.curv_center_lats[i][j][k], self.curv_center_levs[i][j][k])
        if mismatch_found:
            self.fail("data mismatch found for bilinear regridding without flags")

        # Test with only center data and with flags
        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_levs, self.curv_center_ignr)
        regridder.assignCurvField()
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_levs, self.rect_center_ignr)
        regridder.assignRectField(self.rect_data)
        regrid_data = regridder.regridRectToCurv(self.undef_val,
                                                 ESMP.ESMP_REGRIDMETHOD_BILINEAR)
        expect_data = numpy.array(self.curv_data, dtype=numpy.float64)
        undef_flags = numpy.array(self.rect_center_ignr, dtype=numpy.bool)
        expect_data[undef_flags] = self.undef_val
        mismatch_found = False
        for i in xrange(margin, expect_data.shape[0] - margin):
            for j in xrange(margin, expect_data.shape[1] - margin):
                for k in xrange(margin, expect_data.shape[2] - margin):
                    if numpy.abs(expect_data[i, j, k] - regrid_data[i, j, k]) > delta:
                        mismatch_found = True
                        print "expect = %#6.4f, found = %#6.4f for lon = %5.1f, " \
                              "lat = %5.1f, lev = %5.1f" % (expect_data[i, j, k], 
                              regrid_data[i, j, k], self.curv_center_lons[i][j][k], 
                              self.curv_center_lats[i][j][k], self.curv_center_levs[i][j][k])
        if mismatch_found:
            self.fail("data mismatch found for bilinear regridding with flags")


    def test10RegridRectToCurvConserve(self):
        '''
        Tests the CurvRect3DRegridder.regridRectToCurv method.
        '''
        margin = 2
        delta = 1.0

        regridder = CurvRect3DRegridder()

        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_levs, None, True,
                                 self.curv_corner_lons, self.curv_corner_lats,
                                 self.curv_corner_levs)
        regridder.assignCurvField()
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_levs, None, True,
                                 self.rect_corner_lons, self.rect_corner_lats,
                                 self.rect_corner_levs)
        regridder.assignRectField(self.rect_data)
        regrid_data = regridder.regridRectToCurv(self.undef_val,
                                                 ESMP.ESMP_REGRIDMETHOD_CONSERVE)
        expect_data = numpy.array(self.curv_data, dtype=numpy.float64)
        mismatch_found = False
        for i in xrange(margin, expect_data.shape[0] - margin):
            for j in xrange(margin, expect_data.shape[1] - margin):
                for k in xrange(margin, expect_data.shape[2] - margin):
                    if numpy.abs(expect_data[i, j, k] - regrid_data[i, j, k]) > delta:
                        mismatch_found = True
                        print "expect = %#6.4f, found = %#6.4f for lon = %5.1f, " \
                              "lat = %5.1f, lev = %5.1f" % (expect_data[i, j, k], 
                              regrid_data[i, j, k], self.curv_center_lons[i][j][k], 
                              self.curv_center_lats[i][j][k], self.curv_center_levs[i][j][k])
        if mismatch_found:
            self.fail("data mismatch found for conservative regridding")


    def test11RegridRectToCurvPatch(self):
        '''
        Tests the CurvRect3DRegridder.regridRectToCurv method.
        '''
        margin = 2
        delta = 1.0

        # Mark as the last test so ESMPControl().stopESMP will be called
        self.last_test = True

        regridder = CurvRect3DRegridder()

        regridder.createCurvGrid(self.curv_center_lons, self.curv_center_lats,
                                 self.curv_center_levs)
        regridder.assignCurvField()
        regridder.createRectGrid(self.rect_center_lons, self.rect_center_lats,
                                 self.rect_center_levs)
        regridder.assignRectField(self.rect_data)
        regrid_data = regridder.regridRectToCurv(self.undef_val,
                                                 ESMP.ESMP_REGRIDMETHOD_PATCH)
        expect_data = numpy.array(self.curv_data, dtype=numpy.float64)
        mismatch_found = False
        for i in xrange(margin, expect_data.shape[0] - margin):
            for j in xrange(margin, expect_data.shape[1] - margin):
                for k in xrange(margin, expect_data.shape[2] - margin):
                    if numpy.abs(expect_data[i, j, k] - regrid_data[i, j, k]) > delta:
                        mismatch_found = True
                        print "expect = %#6.4f, found = %#6.4f for lon = %5.1f, " \
                              "lat = %5.1f, lev = %5.1f" % (expect_data[i, j, k], 
                              regrid_data[i, j, k], self.curv_center_lons[i][j][k], 
                              self.curv_center_lats[i][j][k], self.curv_center_levs[i][j][k])
        if mismatch_found:
            self.fail("data mismatch found for patch regridding")


    def tearDown(self):
        '''
        Finalize ESMP if it has been initialized and if this is the last test
        '''
        if self.last_test:
            ESMPControl().stopESMP(False)


if __name__ == "__main__":
    '''
    Run the unit tests in this module.
    '''
    unittest.main()

