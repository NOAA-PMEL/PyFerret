	INTEGER FUNCTION UVAR_NUMBER( name_code )

* find the number of the given user-defined variable

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/19/87
* revision 0.1 - 8/29/87 - added vms_str_success
* Ultrix/RISC port - 2/91 - removed VMS runtime library reference
*			    (STR_CASE_BLIND_COMPARE is in the TMAP library)

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/gfdl_vms.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:GFDL_VMS.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	CHARACTER*(*)	name_code

* internal variable declarations:
	INTEGER	STR_CASE_BLIND_COMPARE, vax_code, uvar

* locate the requested variable in the uvar variables
	DO 100 uvar = 1, max_uvar
	   vax_code = STR_CASE_BLIND_COMPARE( name_code, 
     .                       uvar_name_code(uvar) )
	   IF ( vax_code .EQ. vms_str_success
     .	  .AND. uvar_num_items( uvar ) .NE. uvar_deleted ) THEN
	      UVAR_NUMBER = uvar
	      RETURN
	   ENDIF
 100	CONTINUE

* not found
	UVAR_NUMBER = unspecified_int4
	RETURN
	END
