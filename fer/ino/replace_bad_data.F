	SUBROUTINE REPLACE_BAD_DATA( src, msrc, cx, old_bad )

* replace all occurances of the value "old_bad" with the bad data flag
* associated with this msrc

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/19/87
* V200:  5/18/89 - 4D-symmetrical version
*	10/16/89 - modified array declarations using REPLACE_BAD_DATA_SUB

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	msrc, cx
	REAL    src( * )
	REAL	old_bad

	CALL REPLACE_BAD_DATA_SUB( cx, old_bad,
     .			    src, mr_lo_s1(msrc), mr_hi_s1(msrc),
     .				 mr_lo_s2(msrc), mr_hi_s2(msrc),
     .				 mr_lo_s3(msrc), mr_hi_s3(msrc),
     .				 mr_lo_s4(msrc), mr_hi_s4(msrc),
     .				 mr_bad_data( msrc )		 )

	RETURN
	END

