''' Represents a Ferret variable that is part of a forecast-model-run collection '''

from __future__ import print_function

import pyferret

_fmrc_var_qualifier = '__new_fmrc_variable__'

class FerFMRCVar(pyferret.FerVar):
    '''
    FerVar that is part of a FerFMRC (forecast-model-run collection dataset).
    Such a Ferret variable has restrictions on the time and forecast axes,
    and has special transformations to better view the quality of the forecast 
    represented by this variable.
    '''

    def __init__(self, qual):
        '''
        Creates a FerFMRCVar, which is a FerVar that is part of a FerFMRC 
        (forecast-model-run collection dataset).  FerFMRCVar objects should 
        only be created by an FerFMRC object; this constructor should never 
        be called directly.
        '''
        super(FerFMRCVar, self).__init__()
        if not qual == __fmrc_var_qualifier:
           raise  ValueError('FerFMRCVar objects should only be created by FerFMRC dataset objects')

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
        diagdefn = '%s[d=%s,gt(TF_TIMES[d=%s])=TF_CAL_T[d=%s]]' % \
                   (self._varname, self._dsetname, self._dsetname, self._dsetname)
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
        skilldefn = '%s[d=%s,gt(TF_TIMES[d=%s])=TF_CAL_T[d=%s],gf(TF_TIMES[d=%s])=TF_LAG_F[d=%s]]' % \
                   (self._varname, self._dsetname, self._dsetname, self._dsetname, self._dsetname, self._dsetname)
        skillvar = pyferret.FerVar(defn=skilldefn, title=self._title)
        skillvar._requires.add(self._varname.upper())
        skillvar._requires.add("TF_TIMES")
        return skillvar

