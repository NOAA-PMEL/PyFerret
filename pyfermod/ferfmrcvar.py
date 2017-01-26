''' Represents a Ferret variable that is part of a forecast-model-run collection '''

import pyferret

_fmrc_var_qualifier = '__new_fmrc_variable__'

class FerFMRCVar(pyferret.FerVar):
    '''
    A Ferret variable (and thus, a FerVar) that is part of a FerFMRCDSet (forecast-
    model-run collection dataset).  Such a Ferret variable has restrictions on the 
    time and forecast axes, and has special transformations to better view the 
    quality of the forecast represented by this variable.
    '''

    def __init__(self, fvar, qual):
        '''
        Create from the given FerVar that is part of a FerFMRCDSet (forecast-model-run 
        collection dataset).  FerFMRCVar objects should only be created by a FerFMRCDSet 
        object; this constructor should never be called directly.  The FerFMRCVar object
        returned should replace the given FerVar.
        '''
        super(FerFMRCVar, self).__init__(defn=fvar._definition, title=fvar._title)
        if (qual != _fmrc_var_qualifier) or (not fvar._varname) or \
           (not fvar._dsetname) or (not fvar._definition) or (not fvar._isfilevar):
            raise  ValueError('FerFMRCVar objects should only be created by FerFMRCDSet objects')
        self._markasknownvar(fvar._varname, fvar._dsetname, True)

    def __repr__(self):
        '''
        Representation of this FerFMRCVar
        '''
        infostr = "FerFMRCVar(varname='%s', dsetname='%s', title = '%s', defn='%s')" \
                  % (self._varname, self._dsetname, self._title, self._definition)
        return infostr

    # Note: Just use the __cmp__, __eq__, and __ne__ methods inherited from FerVar.
    #       No need to make these orbjects appear to be different from a FerVar.

    def diagform(self):
        '''
        Returns an anonymous FerVar that is the transformation of this FerFMRCVar 
        to a "diagonal" form, where the T (time) axis is the date forecasted and 
        the F (forecast) axis is the date the forecast was made.
        '''
        if (not self._varname) or (not self._dsetname):
            raise ValueError('Invalid FerFMRCVar object')
        # TF_TIMES is an automatically generated variable for FMRC datasets in Ferret
        # TF_CAL_T is an automatically generated axis for FMRC datasets in Ferret
        diagdefn = '%s[d=%s,gt(TF_TIMES[d=%s])=TF_CAL_T]' % \
                   (self._varname, self._dsetname, self._dsetname)
        diagvar = pyferret.FerVar(defn=diagdefn, title=self._title)
        diagvar._requires.add(self._varname.upper())
        diagvar._requires.add("TF_TIMES")
        return diagvar

    def skillform(self):
        '''
        Returns an anonymous FerVar that is the transformation of this FerFMRCVar 
        to the "skill" form, where the T (time) axis is the date forecasted and 
        the F (forecast) axis is the lead time for the date forecasted (forecasted
        time minus time that the forecast was made).
        '''
        if (not self._varname) or (not self._dsetname):
            raise ValueError('Invalid FerFMRCVar object')
        # TF_TIMES is an automatically generated variable for FMRC datasets in Ferret
        # TF_CAL_T is an automatically generated axis for FMRC datasets in Ferret
        # TF_LAG_F is an automatically generated axis for FMRC datasets in Ferret
        skilldefn = '%s[d=%s,gt(TF_TIMES[d=%s])=TF_CAL_T,gf(TF_TIMES[d=%s])=TF_LAG_F]' % \
                    (self._varname, self._dsetname, self._dsetname, self._dsetname)
        skillvar = pyferret.FerVar(defn=skilldefn, title=self._title)
        skillvar._requires.add(self._varname.upper())
        skillvar._requires.add("TF_TIMES")
        return skillvar

