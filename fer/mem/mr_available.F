	SUBROUTINE MR_AVAILABLE ( mr )

* flag that the specified variable is available to be used as a component and
* to be deleted should memory space be tight

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

* set flag that it is available for use and for deletion
	mr_protected( mr ) = mr_not_protected

! temp diag
	IF (mr_del_flink(mr) .NE. unspecified_int4) 
     .		WRITE (6,*)' mr_avail_err'

* insert the variable at the head of the deletion priority chain
	last_head = mr_del_flink(0)
	mr_del_blink(mr) = 0
	mr_del_flink(mr) = last_head
	mr_del_flink(0)  = mr
	mr_del_blink(last_head) = mr

	RETURN
	END

