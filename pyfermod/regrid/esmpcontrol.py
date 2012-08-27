'''
Singleton class for starting and stopping ESMP (using ESMP.ESMP_Intialize
and ESMP.ESMP_Finalize) once, and only once, in a Python session.

@author: ksmith
'''

import os
import atexit
import ESMP

class ESMPControl(object):
    '''
    This singleton class and its pair of methods startCheckESMP
    and stopESMP are designed to called ESMP.ESMP_Initialize and
    ESMP.ESMP_Finalize once, and only once, in a Python session.

    When ESMP is initialized in the first call to startCheckESMP,
    stopESMP is registered with atexit to ensure ESMP is always
    finalized prior to exiting Python.  If startCheckESMP was
    called, and then stopESMP was called, any subsequent calls to
    startCheckESMP will return False to prevent reinitialization,
    which currently causes problems in ESMP.
    '''

    # The singleton instance for this class
    __singleton = None

    def __new__(cls):
        '''
        Returns the singleton instance of this class,
        creating it if it does not already exist.
        '''
        # If this is the first call, create the singleton object 
        # and initialize its attributes.
        if cls.__singleton == None:
            cls.__singleton = super(ESMPControl, cls).__new__(cls)
            cls.__singleton.__esmp_initialized = False
            cls.__singleton.__esmp_finalized = False
        return cls.__singleton


    def startCheckESMP(self):
        '''
        Calls ESMP.ESMP_Initialize and registers stopESMP with atexit
        when called the first time.  Subsequent calls only return
        whether or not ESMP is initialized.  Registering stopESMP with
        atexit ensures the ESMP.ESMP_Finalize will always be called
        prior to exiting Python.

        Arguments:
            None
        Returns:
            True  - if ESMP_Initialize has been called, either as a
                    result of this call or from a previous call
            False - if stopESMP has called ESMP_Finalize
        '''
        # Return False if attempting to re-initialize
        if self.__esmp_finalized:
            return False
        # Call ESMP_Initialize if not already done previously 
        if not self.__esmp_initialized:
            ESMP.ESMP_Initialize()
            atexit.register(self.stopESMP)
            self.__esmp_initialized = True
        # Return True if ESMP_Intialize had already or has now been called
        return True


    def stopESMP(self, delete_log=False):
        '''
        Calls ESMP.ESMP_Finalize if ESMP had been initialzed using
        startCheckESMP.  This function is registered with atexit
        by startCheckESMP when it calls ESMP.ESMP_Initialize to
        ensure ESMP.ESMP_Finalize is always called before exiting
        Python.  However, this function can be called directly
        (presumably to delete the log file) before exiting without
        causing problems.

        Arguments:
            delete_log: if True, the ESMF log file 'PET0.ESMF_LogFile'
                        will be deleted after ESMP.ESMP_Finalize is
                        called.  Any failure to delete the log file
                        is silently ignored.
        Returns:
            None
        '''
        # If ESMP not initialize, or already finalized, just return
        if not self.__esmp_initialized:
            return
        if self.__esmp_finalized:
            return
        # Call ESMP_Finalize and set flags indicating this has been done
        ESMP.ESMP_Finalize()
        self.__esmp_initialized = False
        self.__esmp_finalized = True
        # Optionally delete the ESMF log file
        if delete_log:
            try:
                os.remove('PET0.ESMF_LogFile')
            except Exception:
                pass

