'''
Representation of a space-time region of interest
'''

import pyferret

class FerRegion(object)
    '''
    A logically rectagular region specified using axis limit qualifiers.
    Also used as a superclass for more elaborate regions to provide a
    bounding box for the more elaborate region.
    '''

    def __init__(self, X=None, Y=None, Z=None, T=None, E=None, F=None,
                       I=None, J=None, K=None, L=None, M=None, N=None, qual=''):
        '''
        A logically rectagular regions specified using axis limit qualifiers.
        (eg; X="90W:20E",Y="20N:60N").  

        The X, Y, Z, T, E, and F arguments refer to an axis coordinate value 
        or range of axis coordinate values.  Values given as strings are 
        interpreted according to Ferret syntax rules (ranges can include the
        endpoint).
            X (float, str, float slice, or str slice): X (longitude) axis coordinate position or range
            Y (float, str, float slice, or str slice): Y (latitude) axis coordinate position or range
            Z (float, str, float slice, or str slice): Z (level) axis coordinate position or range
            T (float, str, float slice, or str slice): T (time) axis coordinate position or range
            E (float, str, float slice, or str slice): E (ensemble) axis coordinate position or range
            F (float, str, float slice, or str slice): F (forecast) axis coordinate position or range

        The I, J, K, L, M, and N arguments refer to an axis coordinate indices 
        or range of values.  Indices given as integers are interpreted as 
        python indices (starts with zero, excludes endpoint in ranges).  
        Values given as strings are interpreted according to Ferret syntax 
        rules (starts at one, includes endpoint in ranges).
            I (int, str, int slice, or str slice): X (longitude) axis index or range of indices
            J (int, str, int slice, or str slice): Y (latitude) axis index or range of indices
            K (int, str, int slice, or str slice): Z (level) axis index or range of indices
            L (int, str, int slice, or str slice): T (time) axis index or range of indices
            M (int, str, int slice, or str slice): E (ensemble) axis index or range of indices
            N (int, str, int slice, or str slice): F (forecast) axis index or range of indices

        For any axis, either a coordinate or an index specification can be
        given, but not both.
        '''
        if (X is not None) and (I is not None):
            raise ValueError('X and I cannot both be given')
        if (Y is not None) and (J is not None):
            raise ValueError('Y and J cannot both be given')
        if (Z is not None) and (K is not None):
            raise ValueError('Z and K cannot both be given')
        if (T is not None) and (L is not None):
            raise ValueError('T and L cannot both be given')
        if (E is not None) and (M is not None):
            raise ValueError('E and M cannot both be given')
        if (F is not None) and (N is not None):
            raise ValueError('F and N cannot both be given')

        self._mask = None
        self._qualifiers = ''

        self._qualifiers += _interpcoord('X', X)
        self._qualifiers += _interpcoord('Y', Y)
        self._qualifiers += _interpcoord('Z', Z)
        self._qualifiers += _interpcoord('T', T)
        self._qualifiers += _interpcoord('E', E)
        self._qualifiers += _interpcoord('F', F)
        self._qualifiers += _interpcoord('F', F)

        self._qualifiers += _interpindex('I', I)
        self._qualifiers += _interpindex('J', J)
        self._qualifiers += _interpindex('K', K)
        self._qualifiers += _interpindex('L', L)
        self._qualifiers += _interpindex('M', M)
        self._qualifiers += _interpindex('N', N)

        if not isinstance(qual, str):
            raise ValueError('qual (Ferret qualifiers) must be a string')
        if qual:
            self._qualifiers += qual


     def __str__(self):
         return 'FerRegion(qual="' + self._qualifiers + '")'


     @staticmethod
     def _interpcoord(name, value):
         '''
         Converts a coordinate value or range specification to a Ferret coordinate qualifier
             name ('X','Y','Z','T','E','F'): name of the value for the qualifier
             value (number, str, number slice, or str slice): value to interpret
         Returns (str): Ferret coordinate qualifier ('/X=165')
         If value is None, name is ignored and an empty string is returned.
         Raises ValueError if unable to interpret value.
         '''
         if value is None:
             return ''
         if not name in ('X','Y','Z','T','E','F'):
             raise ValueError('Invalid variable name "' + str(name) + '"')
         qual = '/' + name + '='
         if isinstance(value, numbers.Real):
             return qual + str(value)
         elif isinstance(value, str):
             val = value.strip()
             if val:
                 return qual + val
         elif isinstance(value, slice):
             start = value.start
             stop = value.stop
             step = value.step
             if isinstance(start, numbers.Real) and isinstance(stop, numbers.Real) and (step is None):
                 return qual + str(start) + ':' + str(stop)
             elif isinstance(start, numbers.Real) and isinstance(stop, numbers.Real) and isinstance(step, numbers.Real):
                 return qual + str(start) + ':' + str(stop) + ':' + str(step)
             elif isinstance(start, str) and isinstance(stop, str) and (step is None):
                 start = start.strip()
                 stop = stop.strip()
                 if start and stop:
                     return qual + start + ':' + stop
             elif isinstance(start, str) and isinstance(stop, str) and isinstance(step, str):
                 start = start.strip()
                 stop = stop.strip()
                 step = step.strip()
                 if start and stop and step:
                     return qual + start + ':' + stop + ':' + step
         raise ValueError('definition for ' + name + ' is invalid')


     @staticmethod
     def _interpindex(name, value):
         '''
         Converts a coordinate index or range specification to a Ferret coordinate qualifier
         Int values and slices are interpreted as Python syntax (start at zero, ranges exclude
         the endpoint).  String values and slices are interpreted as Ferret syntax (start at
         one, ranges include endpoint).
             name ('I','J','K','L','M','N'): name of the value for the qualifier
             value (int, str, int slice, or str slice): value to interpret
         Returns (str): Ferret coordinate qualifier ('/I=1:10')
         Raises ValueError if unable to interpret value.
         '''
         if value is None:
             return ''
         if not name in ('I','J','K','L','M','N'):
             raise ValueError('Invalid variable name "' + str(name) + '"')
         qual = '/' + name + '='
         if isinstance(value, numbers.Integral)
             return qual + str(value + 1)
         elif isinstance(value, str):
             val = value.strip()
             if val:
                 return qual + val
         elif isinstance(value, slice):
             start = value.start
             stop = value.stop
             step = value.step
             if isinstance(start, numbers.Integral) and isinstance(stop, numbers.Integral) and (step is None):
                 return qual + str(start + 1) + ':' + str(stop)
             elif isinstance(start, numbers.Integral) and isinstance(stop, numbers.Integral) and isinstance(step, numbers.Integral):
                 return qual + str(start + 1) + ':' + str(stop) + ':' + str(step)
             elif isinstance(start, str) and isinstance(stop, str) and (step is None):
                 start = start.strip()
                 stop = stop.strip()
                 if start and stop:
                     return qual + start + ':' + stop
             elif isinstance(start, str) and isinstance(stop, str) and isinstance(step, str):
                 start = start.strip()
                 stop = stop.strip()
                 step = step.strip()
                 if start and stop and step:
                     return qual + start + ':' + stop + ':' + step
         raise ValueError('definition for ' + name + ' is invalid')


