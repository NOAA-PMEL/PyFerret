	SUBROUTINE COPY_2D_TO_3D ( src, msrc, dst, mdst )

* copy data from a 2D (XY and T) grid to a 3D grid and then replicate the
* data along the Z axis of the dstination

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  8/1/89 - 4D symmetrical
*	10/11/89 - use ..._SUB - avoid adjustable dimensioning with array

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER msrc, mdst
	REAL    src( * ), dst( * )

* --- end of introductory code ---

	CALL COPY_2D_TO_3D_SUB(
     .			    src, mr_lo_s1(msrc), mr_hi_s1(msrc),
     .				 mr_lo_s2(msrc), mr_hi_s2(msrc),
     .
     .				 mr_lo_s4(msrc), mr_hi_s4(msrc),
     .			    dst, mr_lo_s1(mdst), mr_hi_s1(mdst),
     .				 mr_lo_s2(mdst), mr_hi_s2(mdst),
     .				 mr_lo_s3(mdst), mr_hi_s3(mdst),
     .				 mr_lo_s4(mdst), mr_hi_s4(mdst)	)


	RETURN
	END 
