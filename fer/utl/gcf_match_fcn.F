	INTEGER FUNCTION GCF_MATCH_FCN( template )

* return the ID number of the first grid changing function whose name
* matches the template

* V450: 12/96

* calling argument declarations
	CHARACTER template*(*)

* internal variable declarations
	LOGICAL  MATCH_TEMPLATE
	INTEGER  i

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xgrid_chg_fcns.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XGRID_CHG_FCNS.CMN'
#endif

* check the internal grid changing functions first
	DO 100 i = 1, gfcn_num_internal
	   IF ( MATCH_TEMPLATE(gfcn_name(i),template) ) THEN
	      GCF_MATCH_FCN = i
	      RETURN
	   ENDIF
 100	CONTINUE

* no match. so check external functions
	CONTINUE   ! not yet implemented

* no luck -- return value indicating failure
	GCF_MATCH_FCN = unspecified_int4
	RETURN

	END
