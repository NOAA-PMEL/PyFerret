#! python
#

'''
Regridders designed for use in PyFerret, especially for Python external 
functions for PyFerret.  Includes the singleton class ESMPControl to 
safely start and stop ESMP once, and only once, in a Python session.

@author: Karl Smith
'''

# Import classes given in modules in this package so they are all seen here.
try:
    from esmpcontrol import ESMPControl
    from regrid2d import CurvRectRegridder
except ImportError:
    # No ESMP, but do not raise an error until attempting to actually use it
    pass


