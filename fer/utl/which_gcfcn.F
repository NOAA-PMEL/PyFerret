	INTEGER FUNCTION WHICH_GCFCN( uvar )

* which grid-changing fuction is used by the indicated GC variable?

* V450: 1/97

* calling argument declarations
	INTEGER	uvar

* internal variable declarations
	INTEGER GCF_FIND_FCN, item

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif


* in reverse Polish order the final item of a grid changing variable will
* always be a grid changing function
	item = uvar_num_items(uvar)

* identify the function
	WHICH_GCFCN = GCF_FIND_FCN( uvar_text(uvar)
     .					(uvar_item_start(item,uvar)
     .					:uvar_item_end  (item,uvar)) )

	RETURN
	END
