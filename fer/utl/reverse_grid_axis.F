	SUBROUTINE REVERSE_GRID_AXIS ( axis, cx, mr, dat )

* reverse the ordering of the data along the indicated axis

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V320:  2/16/95

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
	INTEGER axis, cx, mr
	REAL    dat( * )

	CALL REVERSE_GRID_AXIS_SUB( axis, cx, dat,
     .			         mr_lo_s1(mr), mr_hi_s1(mr),
     .				 mr_lo_s2(mr), mr_hi_s2(mr),
     .				 mr_lo_s3(mr), mr_hi_s3(mr),
     .				 mr_lo_s4(mr), mr_hi_s4(mr) )

	RETURN
	END 
