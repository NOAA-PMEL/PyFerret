	SUBROUTINE CREATE_NEW_CONTEXT( source_cx, new_cx, status )

* push context stack to allocate space for a new context and copy the source
* context into the new slot as default

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/23/87

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER		source_cx, new_cx, status

* push the context stack
	CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) RETURN
	new_cx = cx_stack_ptr

* set up default values
	CALL TRANSFER_CONTEXT( source_cx, new_cx )

	RETURN
	END
