	SUBROUTINE DELETE_OLD_EXPR

* delete from memory all "expression" variables - internally defined user-vars
* that were created by expressions given with the commmand line

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 10/31/86
* revision 0.1 - 1/30/87 - modified for mr_protected declared as INTEGER
* revision 1.0 - 2/23/87 - changes for conversion to INTERPRETATION STACK using
*			   the idea that all expressions are special cases of
*			   user-defined variables
* revision 1.1 - 5/6/87  - corrected bug in UVAR_NUMBER (status) call
* V200:   7/7/89 - expression may have been deleted outside this routine
* 	 2/19/90 - major rewrite: loop through ALL user vars deleting "EX#"

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* internal variable declarations:
	INTEGER	uvar

* eliminate all user variables created by the last expression given
	DO 100 uvar = 1, max_uvar
	   IF ( uvar_num_items( uvar ) .EQ. uvar_deleted ) GOTO 100
	   IF ( uvar_name_code(uvar)(:3) .EQ. 'EX#' )
     .					CALL DELETE_USER_VAR( uvar )
 100	CONTINUE

	RETURN
	END

