	SUBROUTINE STACK_PTR_UP ( stack_ptr, max_stack, status )

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 10/14/86
* revision 0.1 - 5/22/87  - stack pointer is only incremented for non-error
*			    calls to avoid subscript out of bounds in calling
*			    routine

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
#endif

	INTEGER	stack_ptr, max_stack, status

	IF ( stack_ptr .GE. max_stack ) CALL ERRMSG
     .		( ferr_stack_ovfl, status, ' ',*5000 )

	stack_ptr = stack_ptr + 1

	status = ferr_ok
	RETURN

* error exit
 5000	RETURN
	END
