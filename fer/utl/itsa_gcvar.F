	LOGICAL FUNCTION ITSA_GCVAR( uvar )

* is the indicated variable one which is contains a grid changing function?

* V450: 1/97

* calling argument declarations
	INTEGER	uvar

* internal variable declarations

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif


* in reverse Polish order the final item of a grid changing variable will
* always be a grid changing function

	ITSA_GCVAR = uvar_item_type( uvar_num_items(uvar), uvar )
     .		.EQ. alg_grid_chg_fcn

	RETURN
	END
