	SUBROUTINE ASSIGN_TEMP_VAR ( cx, start_blk, num_blks, mr, status )

* catalog in memory the description of this temporary variable
* typically an intermediate result in a calculation

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/9/86
* revision 0.1 - 2/4/87 - eliminated need for tvar_grid
* V200:  6/20/89 - variable protection scheme changed

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling arguments and local variables . . . . . . .
	INTEGER	cx, start_blk, num_blks, mr, status

* begin executable code:

* assign the temporary variable into the memory variable list
	CALL ASSIGN_VARIABLE  (	cx,
     .				start_blk,
     .				num_blks,
     .				mr,
     .				status		)
	IF ( status .NE. ferr_ok ) GOTO 5000

* modify assigned slot in memory table to reflect temporary nature of variable
	mr_protected( mr ) = mr_temporary

	RETURN

* error exits
 5000	RETURN
	END
