	INTEGER FUNCTION GCF_NUM_ARGS( ifcn )

* return the number of arguments to this grid changing function
* variable argument functions return a negative value with the
* ABS() equal to the number of **required** arguments

* V450: 12/96

* calling argument declarations
	INTEGER		ifcn

* internal variable declarations

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xgrid_chg_fcns.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XGRID_CHG_FCNS.CMN'
#endif

	IF ( ifcn .GT. gfcn_num_internal ) THEN

	   STOP "GCF_NUM_ARGS - external fcn ??"

	ELSE

	   GCF_NUM_ARGS = gfcn_num_reqd_args(ifcn)

	ENDIF
	RETURN

	END
