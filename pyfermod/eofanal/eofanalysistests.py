'''
Tests for the eofanalysis module.

@author: Karl Smith
'''

import eofanalysis
import math
import numpy
import unittest


class EOFAnalysisTests(unittest.TestCase):
    '''
    Tests of the eofanalysis.EOFAnalysis
    and eofanalysis.InvalidStateError classes
    '''

    def setUp(self):
        '''
        Create some repeatedly used test data
        '''
        self.t_steps   = numpy.arange(0.0, 60.0, 0.25)
        self.cos_t     = numpy.cos(self.t_steps * numpy.pi / 6.0)
        self.sin_t     = numpy.sin(self.t_steps * numpy.pi / 6.0)
        self.csmat     = numpy.matrix([self.cos_t,
                                       self.sin_t + 1.0]).T
        self.ccssmat   = numpy.matrix([self.cos_t * self.cos_t,
                                       self.cos_t * self.sin_t + 1.0,
                                       self.cos_t * self.sin_t + 2.0,
                                       self.sin_t * self.sin_t + 3.0]).T
        self.onetmat   = numpy.matrix([self.cos_t])
        self.onelocmat = self.onetmat.T 
        self.novarmat  = numpy.matrix([self.cos_t, self.cos_t, self.cos_t])


    def test01InvalidStateError(self):
        '''
        Tests initialization of InvalidStateError instances.
        '''
        err = eofanalysis.InvalidStateError("Test case")
        self.assertNotEqual(err, None)


    def test02init(self):
        '''
        Tests initialization of EOFAnalysis instances.
        '''
        # check instantiation from a matrix
        eofanalysis.EOFAnalysis(self.csmat)
        # check instantiation from an numpy.array
        eofanalysis.EOFAnalysis(numpy.array([self.cos_t, 
                                             self.sin_t]).T)
        # check instantiation from a list of lists
        eofanalysis.EOFAnalysis([[1.0, 0.0, 0.0, 0.0],
                                 [0.5, 0.5, 0.5, 0.5],
                                 [0.0, 0.0, 0.0, 1.0]])
        # check instantiation from a string
        eofanalysis.EOFAnalysis("1.0, 0.0, 0.0, 0.0 ; " + \
                                "0.5, 0.5, 0.5, 0.5 ; " + \
                                "0.0, 0.0, 0.0, 1.0")
        # check TypeError raised if spacetimedata is not given
        self.assertRaises(TypeError, eofanalysis.EOFAnalysis)
        # check ValueError raised if spacetime is not valid
        self.assertRaises(ValueError, eofanalysis.EOFAnalysis,
                          numpy.matrix([['a', 'b', 'c'],
                                        ['d', 'e', 'f']]))
        # check UserWarning raised if one time value in each time series
        self.assertRaises(UserWarning, eofanalysis.EOFAnalysis, self.onetmat)
        # no problems if only one location, however
        eofanalysis.EOFAnalysis(self.onelocmat)


    def test03analyze(self):
        '''
        Tests the EOFAnalysis.analyze method.
        '''
        # check a valid case
        onelocanal = eofanalysis.EOFAnalysis(self.onelocmat)
        onelocanal.analyze()
        # check UserError raised if constant time series
        novaranal = eofanalysis.EOFAnalysis(self.novarmat)
        self.assertRaises(UserWarning, novaranal.analyze)


    def test04signiffracs(self):
        '''
        Tests the EOFAnalysis.signiffracs method.
        '''
        # check a trivial case with one location
        onelocanal = eofanalysis.EOFAnalysis(self.onelocmat)
        onelocanal.analyze()
        onelocfracs = onelocanal.signiffracs()
        self.assertTrue(numpy.allclose(onelocfracs, 1.0))
        # check a valid case with all significant EOFs
        csanal = eofanalysis.EOFAnalysis(self.csmat)
        csanal.analyze()
        csfracs = csanal.signiffracs()
        self.assertTrue(numpy.allclose(csfracs, [0.5, 0.5]))
        # check a valid case with some insignificant EOFs
        ccssanal = eofanalysis.EOFAnalysis(self.ccssmat)
        ccssanal.analyze()
        ccssfracs = ccssanal.signiffracs()
        self.assertTrue(numpy.allclose(ccssfracs, [0.5, 0.5, 0.0, 0.0]))
        # check a warned case where no significant EOFs
        novaranal = eofanalysis.EOFAnalysis(self.novarmat)
        try:
            novaranal.analyze()
        except UserWarning:
            pass
        novarfracs = novaranal.signiffracs()
        self.assertTrue(numpy.allclose(novarfracs, 0.0))
        # check InvalidStateError raised if analyze had not been called
        noanal = eofanalysis.EOFAnalysis(self.csmat)
        self.assertRaises(eofanalysis.InvalidStateError, noanal.signiffracs)


    def test05minsignif(self):
        '''
        Test of the EOFAnalysis.minsignif and EOFAnalysis.setminsignif methods.
        '''
        csanal = eofanalysis.EOFAnalysis(self.csmat)
        dfltminsignif = csanal.minsignif()
        # Check the default value
        self.assertAlmostEqual(dfltminsignif, 0.01)
        # Reset the value and check that it took
        csanal.setminsignif(0.05)
        resetminsignif = csanal.minsignif()
        self.assertAlmostEqual(resetminsignif, 0.05)
        # Try resetting to invalid values
        self.assertRaises(ValueError, csanal.setminsignif, 0.00000001)
        self.assertRaises(ValueError, csanal.setminsignif, 0.99999999)


    def test06numeofs(self):
        '''
        Tests the EOFAnalysis.numeofs method.
        '''
        # check a trivial case with one location
        onelocanal = eofanalysis.EOFAnalysis(self.onelocmat)
        onelocanal.analyze()
        self.assertEqual(onelocanal.numeofs(), 1)
        # check a valid case with all significant EOFs
        csanal = eofanalysis.EOFAnalysis(self.csmat)
        csanal.analyze()
        self.assertEqual(csanal.numeofs(), 2)
        # check a valid case with some insignificant EOFs
        ccssanal = eofanalysis.EOFAnalysis(self.ccssmat)
        ccssanal.analyze()
        self.assertEqual(ccssanal.numeofs(), 2)
        # check a warned case where no significant EOFs
        novaranal = eofanalysis.EOFAnalysis(self.novarmat)
        try:
            novaranal.analyze()
        except UserWarning:
            pass
        self.assertEqual(novaranal.numeofs(), 0)
        # check InvalidStateError raised if analyze had not been called
        noanal = eofanalysis.EOFAnalysis(self.csmat)
        self.assertRaises(eofanalysis.InvalidStateError, noanal.numeofs)


    def test07eofvec(self):
        '''
        Tests of the EOFAnalysis.eofvec method.  More extensive tests
        are accomplished in test_datapiece.
        '''
        # check a trivial case
        onelocanal = eofanalysis.EOFAnalysis(self.onelocmat)
        onelocanal.analyze()
        oneloceof1 = onelocanal.eofvec(1)
        self.assertTrue(numpy.allclose(oneloceof1, math.sqrt(0.5)))
        # check a "simple" case
        csanal = eofanalysis.EOFAnalysis(self.csmat)
        csanal.analyze()
        cseof1 = csanal.eofvec(1)
        cseof2 = csanal.eofvec(2)
        if cseof1[0] < 1.0E-10:
            self.assertTrue(numpy.allclose(cseof1, [0.0, math.sqrt(0.5)]))
            self.assertTrue(numpy.allclose(cseof2, [math.sqrt(0.5), 0.0]))
        else:
            self.assertTrue(numpy.allclose(cseof1, [math.sqrt(0.5), 0.0]))
            self.assertTrue(numpy.allclose(cseof2, [0.0, math.sqrt(0.5)]))
        # check a EOF properties of a more complicated example
        ccssanal = eofanalysis.EOFAnalysis(self.ccssmat)
        ccssanal.analyze()
        ccsseof1 = ccssanal.eofvec(1)
        ccsseof2 = ccssanal.eofvec(2)
        self.assertAlmostEqual(numpy.dot(ccsseof1, ccsseof1), 0.25)
        self.assertAlmostEqual(numpy.dot(ccsseof1, ccsseof2), 0.0)
        self.assertAlmostEqual(numpy.dot(ccsseof2, ccsseof2), 0.25)
        # check ValueError raised for invalid EOF numbers
        self.assertRaises(ValueError, ccssanal.eofvec, -1)
        self.assertRaises(ValueError, ccssanal.eofvec, 0)
        self.assertRaises(ValueError, ccssanal.eofvec, 3)
        self.assertRaises(ValueError, ccssanal.eofvec, 5)
        # check InvalidStateError raised if analyze had not been called
        noanal = eofanalysis.EOFAnalysis(self.csmat)
        self.assertRaises(eofanalysis.InvalidStateError, noanal.eofvec, 1)


    def test08tafvec(self):
        '''
        Test of the EOFAnalysis.tafvec method.  More extensive tests
        are accomplished in test_datapiece.
        '''
        # check a trivial case
        onelocanal = eofanalysis.EOFAnalysis(self.onelocmat)
        onelocanal.analyze()
        oneloctaf1 = onelocanal.tafvec(1)
        self.assertTrue(numpy.allclose(oneloctaf1, math.sqrt(2.0) * self.cos_t))
        # check a "simple" case
        csanal = eofanalysis.EOFAnalysis(self.csmat)
        csanal.analyze()
        cstaf1 = csanal.tafvec(1)
        cstaf2 = csanal.tafvec(2)
        if cstaf1[0] < 1.0E-10:
            self.assertTrue(numpy.allclose(cstaf1, math.sqrt(2.0) * self.sin_t))
            self.assertTrue(numpy.allclose(cstaf2, math.sqrt(2.0) * self.cos_t))
        else:
            self.assertTrue(numpy.allclose(cstaf1, math.sqrt(2.0) * self.cos_t))
            self.assertTrue(numpy.allclose(cstaf2, math.sqrt(2.0) * self.sin_t))
        # check a EOF properties of a more complicated example
        ccssanal = eofanalysis.EOFAnalysis(self.ccssmat)
        ccssanal.analyze()
        ccsstaf1 = ccssanal.tafvec(1)
        ccsstaf2 = ccssanal.tafvec(2)
        self.assertAlmostEqual(numpy.dot(ccsstaf1, ccsstaf1), 240.0)
        self.assertAlmostEqual(numpy.dot(ccsstaf1, ccsstaf2), 0.0)
        self.assertAlmostEqual(numpy.dot(ccsstaf2, ccsstaf2), 240.0)
        # check ValueError raised for invalid TAF numbers
        self.assertRaises(ValueError, ccssanal.tafvec, -1)
        self.assertRaises(ValueError, ccssanal.tafvec, 0)
        self.assertRaises(ValueError, ccssanal.tafvec, 3)
        self.assertRaises(ValueError, ccssanal.tafvec, 5)
        # check InvalidStateError raised if analyze had not been called
        noanal = eofanalysis.EOFAnalysis(self.csmat)
        self.assertRaises(eofanalysis.InvalidStateError, noanal.tafvec, 1)

    
    def test09nullvec(self):
        '''
        Test of the EOFAnalysis.nullvec method.
        '''
        ccssanal = eofanalysis.EOFAnalysis(self.ccssmat)
        ccssanal.analyze()
        ccsseof1 = ccssanal.eofvec(1)
        ccsseof2 = ccssanal.eofvec(2)
        ccssnv1 = ccssanal.nullvec(1)
        ccssnv2 = ccssanal.nullvec(2)
        self.assertAlmostEqual(numpy.dot(ccssnv1, ccssnv1), 1.0)
        self.assertAlmostEqual(numpy.dot(ccssnv1, ccssnv2), 0.0)
        self.assertAlmostEqual(numpy.dot(ccssnv2, ccssnv2), 1.0)
        self.assertAlmostEqual(numpy.dot(ccssnv1, ccsseof1), 0.0)
        self.assertAlmostEqual(numpy.dot(ccssnv1, ccsseof2), 0.0)
        self.assertAlmostEqual(numpy.dot(ccssnv2, ccsseof1), 0.0)
        self.assertAlmostEqual(numpy.dot(ccssnv2, ccsseof2), 0.0)
        # check ValueError raised for invalid null vector numbers
        self.assertRaises(ValueError, ccssanal.nullvec, -1)
        self.assertRaises(ValueError, ccssanal.nullvec, 0)
        self.assertRaises(ValueError, ccssanal.nullvec, 3)
        self.assertRaises(ValueError, ccssanal.nullvec, 5)
        # check InvalidStateError raised if analyze had not been called
        noanal = eofanalysis.EOFAnalysis(self.csmat)
        self.assertRaises(eofanalysis.InvalidStateError, noanal.nullvec, 1)


    def test10datapiece(self):
        '''
        Tests of the EOFAnalysis.datapiece method.
        '''
        # Test a trivial example
        onelocanal = eofanalysis.EOFAnalysis(self.onelocmat)
        onelocanal.analyze()
        self.assertTrue(numpy.allclose(onelocanal.datapiece(1),
                                       self.onelocmat))
        # Test the results from a valid example
        ccssanal = eofanalysis.EOFAnalysis(self.ccssmat)
        ccssanal.analyze()
        datatotal = ccssanal.datapiece(0)
        for k in range(1, ccssanal.numeofs()+1):
            eofvec = ccssanal.eofvec(k)
            tafvec = ccssanal.tafvec(k)
            tafeof = numpy.outer(tafvec, eofvec)
            datapiece = ccssanal.datapiece(k)
            self.assertTrue(numpy.allclose(datapiece, tafeof), 
                            "Not True: datapiece(%d) == tafvec(%d).T * eofvec(%d)" % \
                            (k,k,k))
            datatotal += datapiece
        self.assertTrue(numpy.allclose(datatotal, self.ccssmat),
                        "Not True: Sum[k=0->numeofs](datapiece(k)) == OriginalData")
        self.assertTrue(numpy.allclose(ccssanal.datapiece(3), 0.0),
                        "Not True: datapiece of insignificant EOF is insignificant")
        self.assertTrue(numpy.allclose(ccssanal.datapiece(4), 0.0),
                        "Not True: datapiece of insignificant EOF is insignificant")
        # check ValueError raised for invalid EOF numbers
        self.assertRaises(ValueError, ccssanal.datapiece, -1)
        self.assertRaises(ValueError, ccssanal.datapiece, 5)
        # check InvalidStateError raised if analyze had not been called
        noanal = eofanalysis.EOFAnalysis(self.csmat)
        self.assertRaises(eofanalysis.InvalidStateError, noanal.datapiece, 0)


    def test11dataexplained(self):
        '''
        Tests of the EOFAnalysis.dataexplained method.
        '''
        # Test a trivial example
        onelocanal = eofanalysis.EOFAnalysis(self.onelocmat)
        onelocanal.analyze()
        self.assertTrue(numpy.allclose(onelocanal.dataexplained(1),
                                       self.onelocmat))
        # Test the results from a valid example
        ccssanal = eofanalysis.EOFAnalysis(self.ccssmat)
        ccssanal.analyze()
        datatotal = numpy.matrix(numpy.zeros(self.ccssmat.shape))
        for k in range(ccssanal.numeofs()+1):
            datatotal += ccssanal.datapiece(k)
            dataexpld = ccssanal.dataexplained(k)
            self.assertTrue(numpy.allclose(dataexpld, datatotal), 
                "Not True: dataexplained(%d) == Sum[k=0->%d](datapiece(k))" % \
                (k,k))
        self.assertTrue(numpy.allclose(ccssanal.dataexplained(3), self.ccssmat),
                        "Not True: dataexplained of insignif EOF is OriginalData")
        self.assertTrue(numpy.allclose(ccssanal.dataexplained(4), self.ccssmat),
                        "Not True: dataexplained of insignif EOF is OriginalData")
        # check ValueError raised for invalid EOF numbers
        self.assertRaises(ValueError, ccssanal.dataexplained, -1)
        self.assertRaises(ValueError, ccssanal.dataexplained, 5)
        # check InvalidStateError raised if analyze had not been called
        noanal = eofanalysis.EOFAnalysis(self.csmat)
        self.assertRaises(eofanalysis.InvalidStateError, noanal.dataexplained, 0)



if __name__ == "__main__":
    '''
    Run the unit tests in this module.
    '''
    unittest.main()
