	INTEGER FUNCTION FVAR_NUMBER( dset_num, name_code )

* identify the given name_code by finding it among the given data set variables

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/18/87
* revision 0.1 - 8/29/87 - added vms_str_success

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/gfdl_vms.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:GFDL_VMS.PARM'
#endif

* calling argument declarations:
	INTEGER		dset_num
	CHARACTER*(*)	name_code

* internal variable declarations:
	INTEGER		STR_CASE_BLIND_COMPARE, fvar, vax_code

* loop through all file variables
	DO 100 fvar = 1, maxvars
	   IF ( ds_var_setnum( fvar ) .NE. dset_num ) GOTO 100
	   vax_code = STR_CASE_BLIND_COMPARE( name_code, ds_var_code(fvar) )

* found the name code ?
	   IF ( vax_code .EQ. vms_str_success ) THEN
	      FVAR_NUMBER = fvar
	      RETURN
	   ENDIF
 100	CONTINUE

* not found
	FVAR_NUMBER = var_name_nonexist
	RETURN

	END
