	SUBROUTINE STACK_PTR_DN ( stack_ptr, min_stack, status )

* drop stack pointer by 1 and check for error

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 10/14/86
* revision 0.1 - 5/22/87  - stack pointer is only decremented for non-error
*			    calls to avoid subscript out of bounds in calling
*			    routine

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
#endif

	INTEGER	stack_ptr, min_stack, status

	IF ( stack_ptr .LE. min_stack ) CALL ERRMSG
     .		( ferr_stack_undfl, status, ' ',*5000 )

	stack_ptr = stack_ptr - 1

	status = ferr_ok
	RETURN

* error exit
 5000	RETURN
	END
