	INTEGER FUNCTION GCF_FIND_FCN( name )

* return the ID number of a grid changing function based on its name 

* V450: 12/96

* calling argument declarations
	CHARACTER name*(*)

* internal variable declarations
	INTEGER         STR_MATCH

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xgrid_chg_fcns.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XGRID_CHG_FCNS.CMN'
#endif

* check the internal grid changing functions first
	GCF_FIND_FCN = STR_MATCH ( name,
     .                               gfcn_name, gfcn_num_internal )
        IF ( GCF_FIND_FCN .NE. atom_not_found ) THEN

	   RETURN

	ELSE

	   GCF_FIND_FCN = unspecified_int4

	ENDIF	   
	
	RETURN

	END
