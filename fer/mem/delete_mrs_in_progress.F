	SUBROUTINE DELETE_MRS_IN_PROGRESS

* delete from memory all in-progress grids ( grids that have been cataloged in
* the memory variable table but whose computation is not yet completed )
* 6/89 - and other temporary data

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/25/87
* V200:  6/27/89 - delete temporary variables, too 
* V420: 11/95 - also delete un-cached memory vars

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* internal variable declarations:
	INTEGER	mr

	DO 100 mr = 1, max_mrs
	   IF ( mr_protected( mr ) .EQ. mr_in_progress
     .	   .OR. mr_protected( mr ) .EQ. mr_temporary )
     .						CALL DELETE_VARIABLE( mr )
 100	CONTINUE

* also delete un-cached memory variables
	CALL DELETE_UN_CACHED_MVARS

	RETURN
	END

