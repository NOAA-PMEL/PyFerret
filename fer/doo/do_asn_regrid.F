	SUBROUTINE DO_ASN_REGRID( cx_lims, axis, src, msrc, dst, mdst )

* perform a regridding by "association"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V230:  8/22/92

* calling argument declarations:
	INTEGER	cx_lims, axis, msrc, mdst, icoef, itot
	REAL	src(*), dst(*)

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

	CALL ASN_REGRID_1_AXIS( cx_lims, axis,
     .			    cx_lo_ss(cx_lims,axis), cx_hi_ss(cx_lims,axis),
     .			    src, msrc, dst, mdst )

	RETURN
	END	

