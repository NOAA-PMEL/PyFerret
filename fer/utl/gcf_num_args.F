	INTEGER FUNCTION GCF_NUM_ARGS( ifcn )

* return the number of arguments to this grid changing function
* variable argument functions return a negative value with the
* ABS() equal to the number of **required** arguments

* V450: 12/96
*       97.07.16 *jc* adding code to query external functions

* calling argument declarations
	INTEGER		ifcn
	INTEGER efcn_scan, efcn_get_num_reqd_args

* internal variable declarations

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xgrid_chg_fcns.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XGRID_CHG_FCNS.CMN'
#endif

	IF ( ifcn .GT. gfcn_num_internal ) THEN

	   GCF_NUM_ARGS = efcn_get_num_reqd_args(ifcn)

	ELSE

	   GCF_NUM_ARGS = gfcn_num_reqd_args(ifcn)

	ENDIF
	RETURN

	END
