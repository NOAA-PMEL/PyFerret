	INTEGER FUNCTION REGION_NUMBER( reg_name )

* identify the named region

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*

* calling argument declarations:
	CHARACTER*(*)	reg_name

* internal variable declarations:
	INTEGER	STR_CASE_BLIND_COMPARE, ict, vax_code

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/gfdl_vms.parm'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:GFDL_VMS.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* revision 0.0 - 6/25/86

* search the predefined context names
	DO 10 ict = 0, min_context, -1
	   vax_code = STR_CASE_BLIND_COMPARE( reg_name, cx_name( ict ) )

	   IF ( vax_code .EQ. vms_str_success ) THEN
	      REGION_NUMBER = ict
	      RETURN
	   ENDIF

 10	CONTINUE

* given region name doesn't match possible options
	REGION_NUMBER = unspecified_int4
	RETURN

	END
