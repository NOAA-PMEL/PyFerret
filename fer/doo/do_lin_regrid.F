	SUBROUTINE DO_LIN_REGRID( cx_lims, axis, src, msrc, dst, mdst,
     .				  work, icoef, itot )

* call the regridding routine with appropriate work arrays

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/25/88
* V200:  8/3/89 - 4D symmetrical version - regrid only 1 axis

* calling argument declarations:
	INTEGER	cx_lims, axis, msrc, mdst, icoef, itot
	REAL	src(*), dst(*), work(itot)

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

	CALL LIN_REGRID_1_AXIS( cx_lims, axis,
     .			    cx_lo_ss(cx_lims,axis), cx_hi_ss(cx_lims,axis),
     .			    src, msrc, dst, mdst,
     .			    work(1), work(icoef) )

	RETURN
	END	
