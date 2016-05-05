'''
Representation of a space-time region of interest
'''

import numbers
import pyferret

class FerRegion(object):
    '''
    A logically rectagular region specified using axis limit qualifiers.
    Also used as a superclass to provide a bounding box for more elaborate 
    regions.
    '''

    def __init__(self, X=None, Y=None, Z=None, T=None, E=None, F=None,
                       I=None, J=None, K=None, L=None, M=None, N=None, qual=''):
        '''
        A logically rectagular regions specified using axis limit qualifiers;
        eg; X="90W:20E",Y="20N:60N"

        The X, Y, Z, T, E, and F arguments refer to an axis coordinate value 
        or range of axis coordinate values.
        String values are used as-is and thus are interpreted according to 
        Ferret syntax rules (e.g., ranges can include the endpoint).
            X (float, str, or slice of float or str): X (longitude) axis coordinate position or range
            Y (float, str, or slice of float or str): Y (latitude) axis coordinate position or range
            Z (float, str, or slice of float or str): Z (level) axis coordinate position or range
            T (float, str, or slice of float or str): T (time) axis coordinate position or range
            E (float, str, or slice of float or str): E (ensemble) axis coordinate position or range
            F (float, str, or slice of float or str): F (forecast) axis coordinate position or range

        The I, J, K, L, M, and N arguments refer to an axis coordinate index 
        or range of indices.  
        Integer values are interpreted as python indices 
        (starts with zero, excludes endpoint in ranges).  
        String values are used as-is and thus are interpreted according to 
        Ferret syntax rules (e.g., starts at one, includes endpoint in ranges).
            I (int, str, or slice of int or str): X (longitude) axis index or range of indices
            J (int, str, or slice of int or str): Y (latitude) axis index or range of indices
            K (int, str, or slice of int or str): Z (level) axis index or range of indices
            L (int, str, or slice of int or str): T (time) axis index or range of indices
            M (int, str, or slice of int or str): E (ensemble) axis index or range of indices
            N (int, str, or slice of int or str): F (forecast) axis index or range of indices

        For any axis, either a coordinate or an index specification can be
        given, but not both.

        The qual argument is a string giving any additional Ferret-syntax qualifiers.
        '''
        # coordinate qualifiers
        self._xaxisqual = self._interpcoord('X', X)
        self._yaxisqual = self._interpcoord('Y', Y)
        self._zaxisqual = self._interpcoord('Z', Z)
        self._taxisqual = self._interpcoord('T', T)
        self._eaxisqual = self._interpcoord('E', E)
        self._faxisqual = self._interpcoord('F', F)
        # index qualifiers
        idxqual = self._interpindex('I', I)
        if idxqual:
            if self._xaxisqual:
                raise ValueError('X and I cannot both be given')
            self._xaxisqual = idxqual
        idxqual = self._interpindex('J', J)
        if idxqual:
            if self._yaxisqual:
                raise ValueError('Y and J cannot both be given')
            self._yaxisqual = idxqual
        idxqual = self._interpindex('K', K)
        if idxqual:
            if self._zaxisqual:
                raise ValueError('Z and K cannot both be given')
            self._zaxisqual = idxqual
        idxqual = self._interpindex('L', L)
        if idxqual:
            if self._taxisqual:
                raise ValueError('T and L cannot both be given')
            self._taxisqual = idxqual
        idxqual = self._interpindex('M', M)
        if idxqual:
            if self._eaxisqual:
                raise ValueError('E and M cannot both be given')
            self._eaxisqual = idxqual
        idxqual = self._interpindex('N', N)
        if idxqual:
            if self._faxisqual:
                raise ValueError('F and N cannot both be given')
            self._faxisqual = idxqual
        # additional qualifiers
        if not isinstance(qual, str):
            raise ValueError('qual must be a string')
        self._addnlqual = qual.strip()
        # masking variable for more complex regions (for superclasses)
        self._maskvar = None


    def __repr__(self):
        '''
        Representation of this FerRegion.  Arguments are always given in the
        string representation (thus, Ferret syntax for indices and ranges).
        '''
        repstr = 'FerRegion('
        if self._xaxisqual.startswith('/X='):
            repstr += "X='" + self._xaxisqual[3:] + "',"
        elif self._xaxisqual.startswith('/I='):
            repstr += "I='" + self._xaxisqual[3:] + "',"
        if self._yaxisqual.startswith('/Y='):
            repstr += "Y='" + self._yaxisqual[3:] + "',"
        elif self._yaxisqual.startswith('/J='):
            repstr += "J='" + self._yaxisqual[3:] + "',"
        if self._zaxisqual.startswith('/Z='):
            repstr += "Z='" + self._zaxisqual[3:] + "',"
        elif self._zaxisqual.startswith('/K='):
            repstr += "K='" + self._zaxisqual[3:] + "',"
        if self._taxisqual.startswith('/T='):
            repstr += "T='" + self._taxisqual[3:] + "',"
        elif self._taxisqual.startswith('/L='):
            repstr += "L='" + self._taxisqual[3:] + "',"
        if self._eaxisqual.startswith('/E='):
            repstr += "E='" + self._eaxisqual[3:] + "',"
        elif self._eaxisqual.startswith('/M='):
            repstr += "M='" + self._eaxisqual[3:] + "',"
        if self._faxisqual.startswith('/F='):
            repstr += "F='" + self._faxisqual[3:] + "',"
        elif self._faxisqual.startswith('/N='):
            repstr += "N='" + self._faxisqual[3:] + "',"
        if self._addnlqual:
            repstr += "qual='" + self._addnlqual + "'"
        if repstr.endswith(","):
            repstr = repstr[:-1]
        repstr += ")"
        return repstr


    def _ferretqualifierstr(self):
        '''
        Returns the Ferret syntax axis qualifier string describing this region;
        eg, '/X=90W:20E/Y=20N:60N'
        If no axis qualifiers are given, an empty string is returned.
        '''
        qualstr  = self._xaxisqual
        qualstr += self._yaxisqual
        qualstr += self._zaxisqual
        qualstr += self._taxisqual
        qualstr += self._eaxisqual
        qualstr += self._faxisqual
        qualstr += self._addnlqual
        return qualstr


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
        if isinstance(value, numbers.Integral):
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


