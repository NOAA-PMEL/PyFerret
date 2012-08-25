'''
Tests of ESMPControl singleton class

@author: ksmith
'''
import unittest
from esmpcontrol import ESMPControl


class Test(unittest.TestCase):
    '''
    Tests of ESMPControl singleton class
    '''


    def test01ESMPControlNew(self):
        '''
        Test of ESMPControl constructor
        '''
        ctrl1 = ESMPControl()
        ctrl2 = ESMPControl()
        self.assertTrue(ctrl1 is ctrl2, "ESMPControl is not a singleton")


    def test02ESMPControlStartStop(self):
        '''
        Test of ESMPControl.startCheckESMP
        '''
        ctrl = ESMPControl()
        # calling stopESMP before calling startCheckESMP should not cause any problems
        ctrl.stopESMP(False)
        self.assertTrue(ctrl.startCheckESMP(), "First call to startCheckESMP did not return True")
        self.assertTrue(ctrl.startCheckESMP(), "Second call to startCheckESMP did not return True")
        ctrl.stopESMP(True)
        self.assertFalse(ctrl.startCheckESMP(), "Call to startCheckESMP after stopESMP did not return False")
        # another call to stopESMP should not cause any problems
        ctrl.stopESMP(False)
        # creating another instance should have no effect
        ctrl2 = ESMPControl()
        self.assertFalse(ctrl2.startCheckESMP(), "Call to startCheckESMP from a 'new' instance after stopESMP did not return False")


if __name__ == "__main__":
    unittest.main()
