	SUBROUTINE BAD_EDGES( cx, x, mx )

* context cx contains the limits of the valid data within array X
* fill the edges of array X with bad-data flags

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:   8/3/89
*	10/16/89 - modified array declarations via call to *_SUB

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER cx, mx
	REAL    x( * )

* --- end of introductory code ---

	CALL BAD_EDGES_SUB( cx, x,
     .			    mr_lo_s1(mx), mr_hi_s1(mx),
     .			    mr_lo_s2(mx), mr_hi_s2(mx),
     .			    mr_lo_s3(mx), mr_hi_s3(mx),
     .			    mr_lo_s4(mx), mr_hi_s4(mx),
     .			    mr_bad_data(mx) )
	RETURN
	END 

