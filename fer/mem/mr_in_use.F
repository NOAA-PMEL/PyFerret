	SUBROUTINE MR_IN_USE ( mr )

* flag that the specified variable is in use as a component of a calculation
* and is therefore not available to be deleted
* (multiple calculations can claim the same component by incrementing the
* protection counter)

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
	INTEGER		prev, next

* remove it from the deletion chain if it's in there
	IF ( mr_protected(mr) .EQ. mr_not_protected ) THEN
	   prev = mr_del_blink(mr)
	   next = mr_del_flink(mr)
	   mr_del_flink(prev) = next
	   mr_del_blink(next) = prev
	   mr_del_flink(mr)   = unspecified_int4	! temp diag
	   mr_del_blink(mr)   = unspecified_int4	! temp diag
	ENDIF

* bump claims counter
	IF ( mr_protected(mr) .GE. mr_not_protected ) 
     .		mr_protected(mr) = mr_protected(mr) + 1
	   
	RETURN
	END

