	SUBROUTINE SUB_2_GRIDS( plus, mplus, mnus, mmnus, rslt, mrslt )

* subtract 2 grids producing a third

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 6/25/87
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
	INTEGER mplus, mmnus, mrslt
	REAL    plus( * ), mnus( * ), rslt( * )

* --- end of introductory code ---

	CALL SUB_2_GRIDS_SUB(
     .			  plus, mr_lo_s1(mplus), mr_hi_s1(mplus),
     .				mr_lo_s2(mplus), mr_hi_s2(mplus),
     .				mr_lo_s3(mplus), mr_hi_s3(mplus),
     .				mr_lo_s4(mplus), mr_hi_s4(mplus),
     .				mr_bad_data( mplus ),
     .			  mnus, mr_lo_s1(mmnus), mr_hi_s1(mmnus),
     .				mr_lo_s2(mmnus), mr_hi_s2(mmnus),
     .				mr_lo_s3(mmnus), mr_hi_s3(mmnus),
     .				mr_lo_s4(mmnus), mr_hi_s4(mmnus),
     .				mr_bad_data( mmnus ),
     .			  rslt, mr_lo_s1(mrslt), mr_hi_s1(mrslt),
     .				mr_lo_s2(mrslt), mr_hi_s2(mrslt),
     .				mr_lo_s3(mrslt), mr_hi_s3(mrslt),
     .				mr_lo_s4(mrslt), mr_hi_s4(mrslt),
     .				mr_bad_data( mrslt )			)


	RETURN
	END 
