	SUBROUTINE UNPROTECT_ALL_MRS

* eliminate deletion protections from all memory-resident grids

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/25/87
* revision 0.1 - 9/30/87 - added mr_perm_prot
* V200:  6/9/89 - deleted variables flagged in mr_protected

#ifdef unix
	include		'ferret_cmn/ferret.parm'
	include 	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE		'FERRET_CMN:FERRET.PARM'
	INCLUDE 	'FERRET_CMN:XVARIABLES.CMN'
#endif

* internal variable declarations:
	INTEGER	mr

	DO 100 mr = 1, max_mrs
	   IF (mr_protected(mr) .GT. mr_not_protected) 
     .               CALL MR_AVAILABLE( mr )
 100	CONTINUE

	RETURN
	END

