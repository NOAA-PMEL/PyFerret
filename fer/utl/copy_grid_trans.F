	SUBROUTINE COPY_GRID_TRANS ( md1, md2, src, msrc, dst, mdst )

* copy data from one grid to another transposing dim1 and dim2 components
* note: this routine could be eliminated using the more general PERMUTE

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/14/88 - from COPY_GRID ver. 0.0
* V200:	10/11/89 - use *_SUB - avoid adjustable dimensioning with array

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER md1, md2, msrc, mdst
	REAL    src( * ), dst( * )

	CALL COPY_GRID_TRANS_SUB( md1, md2,
     .			    src, mr_lo_s1(msrc), mr_hi_s1(msrc),
     .				 mr_lo_s2(msrc), mr_hi_s2(msrc),
     .				 mr_lo_s3(msrc), mr_hi_s3(msrc),
     .				 mr_lo_s4(msrc), mr_hi_s4(msrc),
     .				 mr_bad_data( msrc ),
     .			    dst, mr_lo_s1(mdst), mr_hi_s1(mdst),
     .				 mr_lo_s2(mdst), mr_hi_s2(mdst),
     .				 mr_lo_s3(mdst), mr_hi_s3(mdst),
     .				 mr_lo_s4(mdst), mr_hi_s4(mdst),
     .				 mr_bad_data( mdst )			)

	RETURN
	END 
