	CHARACTER*(*) FUNCTION FULL_UVAR_NAME(uvar, len)

* append to the given user variable name the data set information
*	var[D=dset_name]

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V402:
* V420:  4/96 - document "(/D default)" on pdset_irrelevant variables

* calling argument declarations:
	INTEGER	uvar

* local variable declarations:
	INTEGER TM_LENSTR1, len, len2

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN_TEXT'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* variable name and definition
	len = TM_LENSTR1(uvar_name_code(uvar))
	FULL_UVAR_NAME = uvar_name_code(uvar)(:len)
	IF (uvar_dset(uvar) .GT. pdset_irrelevant) THEN
	   len2 = TM_LENSTR1(ds_name(uvar_dset(uvar)))
	   FULL_UVAR_NAME = FULL_UVAR_NAME(:len)//'[D='//
     .			ds_name(uvar_dset(uvar))(:len2)//']'
	   len = len + len2 + 4
	ELSEIF (uvar_dset(uvar) .EQ. pdset_irrelevant) THEN
	   FULL_UVAR_NAME = FULL_UVAR_NAME(:len)//' (/D default)'
	   len = len + 13
	ENDIF

	RETURN
	END


