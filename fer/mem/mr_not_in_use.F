	SUBROUTINE MR_NOT_IN_USE ( mr )

* release one claim of being required for calculations from variable mr
* if this release leaves the variable unneeded for any calculations then return
* it to the deletable variable chain

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/12/89

#ifdef unix
	include		'ferret_cmn/ferret.parm'
	include 	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE		'FERRET_CMN:FERRET.PARM'
	INCLUDE 	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER		mr

* internal variable declarations:
	INTEGER		last_head

	IF ( mr_protected(mr) .GT. mr_not_protected )
     .		mr_protected(mr) = mr_protected(mr) - 1


* possibly insert the variable at the head of the deletion priority chain
	IF ( mr_protected(mr) .EQ. mr_not_protected ) THEN
	   last_head = mr_del_flink(0)
	   mr_del_blink(mr) = 0
	   mr_del_flink(mr) = last_head
	   mr_del_flink(0)  = mr
	   mr_del_blink(last_head) = mr
	ENDIF

	RETURN
	END

