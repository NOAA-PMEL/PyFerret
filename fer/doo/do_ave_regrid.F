	SUBROUTINE DO_AVE_REGRID( cx_lims, axis, trans,
     .				  src, msrc, dst, mdst,
     .				    work, itot )

* call the area regridding routine with appropriate work array

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V230:  8/22/92 - based on DO_XYAVE_REGRID

* calling argument declarations:
	INTEGER	cx_lims, axis, trans, msrc, mdst, itot
	REAL	src(*), dst(*), work(itot)

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

	CALL AVE_REGRID_1_AXIS(   cx_lims, axis, trans,
     .			   cx_lo_ss(cx_lims,x_dim), cx_hi_ss(cx_lims,x_dim),
     .			   cx_lo_ss(cx_lims,y_dim), cx_hi_ss(cx_lims,y_dim),
     .			   src, msrc, dst, mdst,
     .			   work(1), itot )

	RETURN
	END	
