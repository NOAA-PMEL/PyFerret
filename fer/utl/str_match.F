	INTEGER FUNCTION STR_MATCH( test, list, n )

* identify the test string by finding it in the given list

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 10/20/87

* calling argument declarations:
	INTEGER		n
	CHARACTER*(*)	test, list(n)

* internal variable declarations:
	INTEGER		STR_CASE_BLIND_COMPARE, i, vax_code

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/gfdl_vms.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:GFDL_VMS.PARM'
#endif

* look through the list of possibilities
	DO 100 i = 1, n
	   vax_code = STR_CASE_BLIND_COMPARE( test, list(i) )
	   IF ( vax_code .EQ. vms_str_success ) THEN
	      STR_MATCH = i
	      RETURN
	   ENDIF
 100	CONTINUE

* no luck - we didn't find it
	STR_MATCH = atom_not_found

	RETURN
	END
