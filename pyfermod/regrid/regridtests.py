'''
Tests of the regrid.__init__ functions

@author: Karl Smith
'''
import unittest
import numpy
from . import __init__ as regrid

class RegridTests(unittest.TestCase):
    '''
    Tests of the regrid.__init__ functions
    '''


    def testQuadCornersFrom3D(self):
        '''
        Test of regrid.quadCornersFrom3D
        '''
        xedges = numpy.linspace(2.0, 10.0, 5)
        yedges = numpy.linspace(15.0, 75.0, 7)
        expect_ptsy, expect_ptsx = numpy.meshgrid(yedges, xedges)

        ptsx3d = numpy.empty((xedges.shape[0] - 1, yedges.shape[0] - 1, 4))
        ptsy3d = numpy.empty((xedges.shape[0] - 1, yedges.shape[0] - 1, 4))

        ptsx3d[:, :, 0] = expect_ptsx[:-1, :-1]
        ptsx3d[:, :, 1] = expect_ptsx[1:, :-1]
        ptsx3d[:, :, 2] = expect_ptsx[1:, 1:]
        ptsx3d[:, :, 3] = expect_ptsx[:-1, 1:]
        ptsy3d[:, :, 0] = expect_ptsy[:-1, :-1]
        ptsy3d[:, :, 1] = expect_ptsy[1:, :-1]
        ptsy3d[:, :, 2] = expect_ptsy[1:, 1:]
        ptsy3d[:, :, 3] = expect_ptsy[:-1, 1:]
        ptsx, ptsy = regrid.quadCornersFrom3D(ptsx3d, ptsy3d)
        self.assertTrue(numpy.allclose(expect_ptsx, ptsx),
                        "Expected X coordinates:\n%s\nFound X coordinates:\n%s" % \
                        (str(expect_ptsx), str(ptsx)))
        self.assertTrue(numpy.allclose(expect_ptsy, ptsy),
                        "Expected Y coordinates:\n%s\nFound Y coordinates:\n%s" % \
                        (str(expect_ptsy), str(ptsy)))

        ptsx3d[:, :, 0] = expect_ptsx[:-1, :-1]
        ptsx3d[:, :, 3] = expect_ptsx[1:, :-1]
        ptsx3d[:, :, 2] = expect_ptsx[1:, 1:]
        ptsx3d[:, :, 1] = expect_ptsx[:-1, 1:]
        ptsy3d[:, :, 0] = expect_ptsy[:-1, :-1]
        ptsy3d[:, :, 3] = expect_ptsy[1:, :-1]
        ptsy3d[:, :, 2] = expect_ptsy[1:, 1:]
        ptsy3d[:, :, 1] = expect_ptsy[:-1, 1:]
        ptsx, ptsy = regrid.quadCornersFrom3D(ptsx3d, ptsy3d)
        self.assertTrue(numpy.allclose(expect_ptsx, ptsx),
                        "Expected X coordinates:\n%s\nFound X coordinates:\n%s" % \
                        (str(expect_ptsx), str(ptsx)))
        self.assertTrue(numpy.allclose(expect_ptsy, ptsy),
                        "Expected Y coordinates:\n%s\nFound Y coordinates:\n%s" % \
                        (str(expect_ptsy), str(ptsy)))


    def testQuadCentroids(self):
        '''
        Test of regrid.quadCentroids
        '''
        xedges = numpy.linspace(2.0, 10.0, 5)
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        yedges = numpy.linspace(15.0, 75.0, 7)
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        ptsx, ptsy = numpy.meshgrid(xedges, yedges)
        expect_ctrx, expect_ctry = numpy.meshgrid(xcenters, ycenters)
        ctrx, ctry = regrid.quadCentroids(ptsx, ptsy)
        self.assertTrue(numpy.allclose(expect_ctrx, ctrx),
                        "Expected X coordinates:\n%s\nFound X coordinates:\n%s" % \
                        (str(expect_ctrx), str(ctrx)))
        self.assertTrue(numpy.allclose(expect_ctry, ctry),
                        "Expected X coordinates:\n%s\nFound X coordinates:\n%s" % \
                        (str(expect_ctry), str(ctry)))

        for k in range(1, ptsx.shape[0]):
            ptsx[k, :] += 3.0 * k
        for k in range(expect_ctrx.shape[0]):
            expect_ctrx[k, :] += 1.5 * (2 * k + 1)
        for k in range(1, ptsy.shape[1]):
            ptsy[:, k] += 4.0 * k
        for k in range(expect_ctry.shape[1]):
            expect_ctry[:, k] += 2.0 * (2 * k + 1)
        ctrx, ctry = regrid.quadCentroids(ptsx, ptsy)
        self.assertTrue(numpy.allclose(expect_ctrx, ctrx),
                        "Expected X coordinates:\n%s\nFound X coordinates:\n%s" % \
                        (str(expect_ctrx), str(ctrx)))
        self.assertTrue(numpy.allclose(expect_ctry, ctry),
                        "Expected X coordinates:\n%s\nFound X coordinates:\n%s" % \
                        (str(expect_ctry), str(ctry)))

        for k in range(1, ptsx.shape[1]):
            ptsx[:, k] += 5.0 * k
        for k in range(expect_ctrx.shape[1]):
            expect_ctrx[:, k] += 2.5 * (2 * k + 1)
        for k in range(1, ptsy.shape[0]):
            ptsy[k, :] += 7.0 * k
        for k in range(expect_ctry.shape[0]):
            expect_ctry[k, :] += 3.5 * (2 * k + 1)
        ctrx, ctry = regrid.quadCentroids(ptsx, ptsy)
        self.assertTrue(numpy.allclose(expect_ctrx, ctrx),
                        "Expected X coordinates:\n%s\nFound X coordinates:\n%s" % \
                        (str(expect_ctrx), str(ctrx)))
        self.assertTrue(numpy.allclose(expect_ctry, ctry),
                        "Expected X coordinates:\n%s\nFound X coordinates:\n%s" % \
                        (str(expect_ctry), str(ctry)))



if __name__ == "__main__":
    unittest.main()
