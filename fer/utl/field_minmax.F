	SUBROUTINE FIELD_MINMAX( arr, mr, cx, lo, hi )

* determine the max and min of a memory within context bounds array

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V450: 

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER mr, cx
	REAL    arr(*), lo, hi

	CALL FIELD_MINMAX_SUB(  arr, cx,
     .				mr_lo_s1(mr), mr_hi_s1(mr),
     .				mr_lo_s2(mr), mr_hi_s2(mr),
     .				mr_lo_s3(mr), mr_hi_s3(mr),
     .				mr_lo_s4(mr), mr_hi_s4(mr),
     .				mr_bad_data(mr), lo, hi )


	RETURN
	END 
