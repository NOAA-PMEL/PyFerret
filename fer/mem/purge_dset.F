	SUBROUTINE PURGE_DSET ( dset )

* delete all memory variables with data from a particular data set

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 10/25/89 - extracted from ERASE_DSET
*	12/18/89 - eliminated the possibility of multiple data set variables

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER	dset

* local variable declarations:
	INTEGER mr

* eliminate memory variables depending on this data set
	DO 100 mr = 1,max_mrs
	   IF ( mr_protected( mr ) .EQ. mr_deleted ) GOTO 100
	   IF ( dset .EQ. mr_data_set(mr) ) CALL DELETE_VARIABLE( mr )
 100	CONTINUE

	RETURN
	END
