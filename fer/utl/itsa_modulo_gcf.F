	LOGICAL FUNCTION ITSA_MODULO_GCF( cx, idim )

* determine whether a particular axis of a grid changing variable should have
* modulo applied to the RESULT of the function rather than to the arguments,
* as is normal for non-GC uvars

* This function ASSUMES that the passed context refers to a grid-changing uvar

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V450: 2/97

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xgrid_chg_fcns.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XGRID_CHG_FCNS.CMN'
#endif

* calling argument declarations:
	INTEGER	cx, idim

* internal variable declarations:
	INTEGER	 WHICH_GCFCN, gcfcn

* get grid-changing function
	gcfcn = WHICH_GCFCN( cx_variable(cx) )

* get axis sources
	ITSA_MODULO_GCF = gfcn_axis_will_be(idim,gcfcn)
     .		     .EQ. pgc_axis_supplied_herein 

	RETURN
	END
