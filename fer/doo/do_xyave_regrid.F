	SUBROUTINE DO_XYAVE_REGRID( cx_lims, src, msrc, dst, mdst,
     .				    work, iwork2, itot )

* call the xy-area regridding routine with appropriate work arrays

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 12/1/89 - 4D symmetrical version - regrid only 1 axis

* calling argument declarations:
	INTEGER	cx_lims, axis, msrc, mdst, iwork2, itot
	REAL	src(*), dst(*), work(itot)

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

	CALL XYAVE_REGRID( cx_lims,
     .			   cx_lo_ss(cx_lims,x_dim), cx_hi_ss(cx_lims,x_dim),
     .			   cx_lo_ss(cx_lims,y_dim), cx_hi_ss(cx_lims,y_dim),
     .			   src, msrc, dst, mdst,
     .			   work(1), work(iwork2) )

	RETURN
	END	
