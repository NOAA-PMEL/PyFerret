'''
Subclass of FerrVar whose data is from an array in Python.
'''

import numbers
import numpy
import pyferret

class FerrPyVar(pyferret.FerrVar):
    '''
    FerrVar whose data is from an array in Python.
    '''

    def __init__(self, grid, data, missval, unit='', descript=''):
        '''
        Create as an anonymous FerrPyVar.  The PyVar representing this data
        will not be assigned in Ferret until this FerrPyVar is assigned a
        name in a dataset.
            grid (FerrGrid): grid for the PyVar
            data (up to 6D array/array-like of float): data for the PyVar
            missval (float or single-element array of float): value used to indicate missing data
            unit (string): unit for the data
            descript (string): description (long name) for the PyVar in Ferret;
                if not given, the Ferret variable name (still to be assigned) will be used.
        '''
        # initialize the FerrVar this object is derived from
        super(FerrPyVar,self).__init__()
        # assign a copy of the grid
        if not isinstance(grid, pyferret.FerrGrid):
            raise ValueError('grid is not a FerrGrid')
        try:
            self._datagrid = grid.copy()
        except (ValueError, TypeError) as ex:
            raise ValueError('grid is invalid: %s' % str(ex))
        # assign a copy of the data
        try:
            self._dataarray = numpy.array(data, dtype=numpy.float64, order='F', copy=True)
        except ValueError:
            raise ValueError('data is not an array or array-like of numbers')
        if self._dataarray.ndim > pyferret.MAX_FERRET_NDIM:
            raise ValueError('data has more than %d dimensions' % pyferret.MAX_FERRET_NDIM)
        # assign the missing value
        try:
            if isinstance(missval, numbers.Real):
                self._missingvalue = numpy.array( (missval,), dtype=numpy.float64)
            else:
                self._missingvalue = numpy.array(missval, dtype=numpy.float64, copy=True)
        except ValueError:
            raise ValueError('missval is not a number or number array')
        if (self._missingvalue.ndim != 1) or (self._missingvalue.shape[0] != 1):
            raise ValueError('missval array has more than one element')
        # assign the data unit
        if unit and not isinstance(unit, str):
            raise ValueError('unit, if given, must be a string')
        if unit:
            self._dataunit = unit
        else:
            self._dataunit = ''
        # assign the variable description
        if descript and not isinstance(descript, str):
            raise ValueError('descript, if given, must be a string')
        if descript:
            self._description = descript
        else:
            self._description = ''

    def copy(self):
        '''
        (overrides FerrVar.copy)
        Returns an anonymous copy (_varname and _datasetname are not copied) 
        of this FerrPyVar.
        '''
        return FerrPyVar(grid=self._datagrid, 
                         data=self._dataarray, 
                         missval=self._missingvalue, 
                         unit=self._dataunit,
                         descript=self._description)

    def __repr__(self):
        '''
        (overrides FerrVar.__repr__)
        Representation of this FerrPyVar
        '''
        infostr = "FerrPyVar(descript='%s', \n" + \
                  "          grid=%s, \n" + \
                  "          data=%s, \n" + \
                  "          missval=%s, \n" + \
                  "          unit='%s')" \
                  % (self._descript, 
                     repr(self._datagrid), 
                     repr(self._dataarray),
                     str(self._missingvalue),
                     self._unit)
        return infostr

    def _assigninferret(self, varname, datasetname):
        '''
        (overrides FerrVar._assigninferret)
        Assign the data in this FerrPyVar as a PyVar in Ferret.
            varname (string): name for the PyVar in Ferret
            datasetname (string): associated the PyVar with this dataset in Ferret
        Raises a ValueError is a problem occurs.
        Note: Ferret will rearrange axes, if necessary, so that any longitude
            axis is the first axis, any latitude axis is the second axis, any
            level axis is the third axis, any time axis is the fourth axis or 
            the sixth axis for a second time axis.  The data will, of course,
            also be appropriately structured so remain consistent with the axes.
        '''
        if not isinstance(varname, str):
            raise ValueError('varname must be a string')
        if not varname:
            raise ValueError('varname is empty')
        if not isinstance(datasetname, str):
            raise ValueError('datasetname must be a string')
        # TODO: fix libpyferret so PyVar's can be created without a dataset
        #       (uses Ferret's dataset '0')
        if not datasetname:
            raise ValueError('a FerrPyVar cannot be associated with an anonymous dataset at this time')
        datadict = { 
            'name': varname,
            'dset': datasetname,
            'data': self._dataarray,
            'missing_value': self._missingvalue,
            'data_unit': self._dataunit,
            'axis_types': self._datagrid._axistypes,
            'axis_coords': self._datagrid._axiscoords,
            'axis_units': self._datagrid._axisunits,
            'axis_names': self._datagrid._axisnames,
        }
        if self._description:
            datadict['title'] = self._description
        try:
            pyferret.putdata(datadict)
        except Exception as ex:
            raise ValueError(str(ex))
        self._varname = varname
        self._datasetname = datasetname
        self._definition = self.ferretname()
        # at this point, the data is copied into Ferret, 
        # so calling clean will not cause any problems

    def _removefromferret(self):
        '''
        Removes (cancels) this PyVar in Ferret, then erases _varname.  
        Raises a ValueError if there is a Ferret problem.  
        This normally is not called by the user; instead delete the 
        FerrPyVar from the dataset.
        '''
        # ignore if this Ferrer PyVar has already been removed from Ferret
        if not self._varname:
            return
        ferrname = self.ferretname()
        cmdstr = 'CANCEL PYVAR %s' % ferrname
        (errval, errmsg) = pyferret.run(cmdstr)
        if errval != pyferret.FERR_OK:
            raise ValueError('unable to remove PyVar %s from Ferret: %s' % (ferrname, errmsg))
        self._varname = ''

