	LOGICAL FUNCTION ITSA_MODULO_GCF( cx, idim )

* determine whether a particular axis of a grid changing variable should have
* modulo applied to the RESULT of the function rather than to the arguments,
* as is normal for non-GC uvars

* This function ASSUMES that the passed context refers to a grid-changing uvar

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V450: 2/97
*       97.07.16 *jc* adding code to query external functions

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
	INTEGER	cx, idim, axis_will_be(4)

* internal variable declarations:
	INTEGER	 WHICH_GCFCN, gcfcn

* get grid-changing function
	gcfcn = WHICH_GCFCN( cx_variable(cx) )

* get axis sources
	IF ( gcfcn .GT. gfcn_num_internal ) THEN

	   call efcn_get_axis_will_be(gcfcn, axis_will_be)
	   ITSA_MODULO_GCF = axis_will_be(idim)
	1	.EQ. pgc_axis_supplied_herein 

	ELSE

	   ITSA_MODULO_GCF = gfcn_axis_will_be(idim,gcfcn)
	1	.EQ. pgc_axis_supplied_herein 

	ENDIF

	RETURN
	END
