'''
Represents Ferret grids in Python.
'''

import numbers
import time
import numpy
import pyferret

_VALID_AXIS_NUMS = frozenset( (pyferret.X_AXIS,
                               pyferret.Y_AXIS,
                               pyferret.Z_AXIS,
                               pyferret.T_AXIS,
                               pyferret.E_AXIS,
                               pyferret.F_AXIS) )

class FerGrid(object):
    '''
    Ferret grid object
    '''

    def __init__(self, name=None, axes=None):
        '''
        Describe a Ferret grid from the given axes
            name (string): Ferret name for the grid or the variable using this grid
            axes (sequence of FerAxis): axes of this grid
        '''
        if name:
            if not isinstance(name, str): 
                raise TypeError('grid name is not a string')
            self._gridname = name.strip()
        else:
            self._gridname = ''
        # axes of this grid
        self._axes = [ None ] * pyferret.MAX_FERRET_NDIM
        if axes:
            try:
                for k in xrange(len(axes)):
                    ax = axes[k]
                    if not isinstance(ax, pyferret.FerAxis):
                        raise ValueError('axes[%d] is not a FerAxis' % k)
                    self._axes[k] = ax.copy()
            except TypeError:
                raise TypeError('axes is not a sequence type')
            except IndexError:
                raise ValueError('more than %d axes specified' % pyferret.MAX_FERRET_NDIM)


    def __repr__(self):
        '''
        Representation to recreate this FerGrid
        '''
        # Not elegant, but will do
        infostr = "FerGrid(name='" + self._gridname + "', axes=[\n"
        for k in xrange(len(self._axes)):
            infostr += '            ' + repr(self._axes[k]) + ',\n'
        infostr += '        ])'
        return infostr


    def __eq__(self, other):
        '''
        Two FerGrids are equal is all their contents are the same.  
        All string values are compared case-insensitive.
        '''
        if not isinstance(other, FerGrid):
            return NotImplemented
        if self._gridname.upper() != other._gridname.upper():
            return False
        if self._axes != other._axes:
            return False
        return True


    def __ne__(self, other):
        '''
        Two FerGrids are not equal is any of their contents are not the same.  
        All string values are compared case-insensitive.
        '''
        if not isinstance(other, FerGrid):
            return NotImplemented
        return not self.__eq__(other)


    def copy(self, name=None, ax=None, newax=None):
        '''
        Returns a copy of this FerGrid object, possibly with one axis replaced.
        The FerGrid object returned does not share any mutable values (namely, 
        the axes) with this FerGrid object.

        name (string): new name for the copied grid.
            If name is given, this will be the name of the new grid.
            If name is not given, then
                if ax, and newax are not given, the name of the grid is also copied;
                otherwise, the name of the new grid is not assigned.

        ax (int): index of the axis to modify; one of 
                pyferret.X_AXIS (0)
                pyferret.Y_AXIS (1)
                pyferret.Z_AXIS (2)
                pyferret.T_AXIS (3)
                pyferret.E_AXIS (4)
                pyferret.F_AXIS (5)
            If ax is not given but newax is given, an attempt is made to replace
            an axis of the same type (longitude, latitude, level, time, custom, abstract).

        newax (FerAxis): new axis to use in the copied grid.
            If newax is not given but ax is given, the indicated axis will be replaced
            by an axis normal to the data (pyferret.AXISTYPE_NORMAL).
        '''
        # figure out the new grid name
        if name:
            newgridname = name
        elif newax or (ax != None):
            newgridname = None
        else:
            newgridname = self._gridname
        # check the index of the axis to replace: 0 - 6, or None
        if ax != None:
            if not ax in _VALID_AXIS_NUMS:
                raise ValueError('ax (%s) is not valid' % str(ax))
        newaxidx = ax
        # check the replacement axis
        if newax:
            if not isinstance(newax, pyferret.FerAxis):
                raise ValueError('newax is not valid (not a FerAxis)')
            if (newaxidx == None) and (newax.gettype() != pyferret.AXISTYPE_NORMAL):
                for k in xrange(len(self._axes)):
                    if self._axes[k].gettype() == newax.gettype():
                        newaxidx = k
                        break
            if newaxidx == None:
                raise ValueError('Unable to determine new axis index from axis type')
        elif newaxidx != None:
            newax = pyferret.FerAxis(axtype=pyferret.AXISTYPE_NORMAL)
        # Create and return the new grid
        newaxes = self._axes[:]
        if newax:
            newaxes[newaxidx] = newax
        newgrid = FerGrid(name=newgridname, axes=newaxes)
        return newgrid


    def getaxes(self):
        '''
        Returns a copy of the list of axes for this grid
        '''
        return self._axes[:]


