	SUBROUTINE DO_XACT_REGRID( cx_lims, axis, src, msrc, dst, mdst,
     .				   work, icoef, itot )

* call the regridding routine with appropriate work arrays

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* revision 0.0 - 10/97

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

	CALL XACT_REGRID_1_AXIS( cx_lims, axis,
     .			    cx_lo_ss(cx_lims,axis), cx_hi_ss(cx_lims,axis),
     .			    src, msrc, dst, mdst, 
     .			    work(1), work(icoef) )

	RETURN
	END	
