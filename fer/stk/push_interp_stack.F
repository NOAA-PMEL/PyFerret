	SUBROUTINE PUSH_INTERP_STACK( level, status )

* push the interpretation stack pointer to the next higher level and copy
* the most common objects to the new level

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/19/87

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER		level, status

* internal variable declarations:
	INTEGER	source_level

* save level from which to copy stack info.
	source_level = level

* push the stack pointer
	CALL STACK_PTR_UP( isp, max_intrp, status )
	IF ( status .NE. ferr_ok ) RETURN

* copy some stack info
	is_obj ( isp ) = is_obj ( source_level )
	is_uvar( isp ) = is_uvar( source_level )
	is_cx  ( isp ) = is_cx  ( source_level )

* ! temp - blank out rest of stack to avoid undetected bugs
	is_act      ( isp ) = unspecified_int4
	is_big_cx   ( isp ) = unspecified_int4
	is_phase    ( isp ) = unspecified_int4
	is_axis     ( isp ) = unspecified_int4
	is_mr       ( isp ) = unspecified_int4
	is_sum      ( isp ) = -777.

	RETURN
	END
