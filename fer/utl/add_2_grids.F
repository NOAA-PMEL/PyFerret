	SUBROUTINE ADD_2_GRIDS( src, msrc, dst, mdst )

* add data from one grid to another

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/30/86
* revision 1.0 - 1/26/87 - rewritten using mr grid descriptions
* V200:  7/26/89 - 4D symmetrical version
*	10/12/89 - use ..._SUB - avoid adjustable dimensioning with array


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

	CALL ADD_2_GRIDS_SUB(
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
