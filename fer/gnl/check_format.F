	SUBROUTINE CHECK_FORMAT( string, status )

* Check a FORTRAN FORMAT string for superficial syntax errors

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V301 1/94 - extracted from EQUAL_FORMAT
* V320 8/94 *kob* -IBM port - had to use risc buff for string concats

* calling argument declarations
	INTEGER		status
	CHARACTER*(*)	string

* internal variable declarations
        LOGICAL TM_DIGIT
	INTEGER	left_paren, right_paren, integer_form

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
!	include 'ferret_cmn/xprog_state.cmn'
#ifdef NO_PASSED_CONCAT
	include 'ferret_cmn/xrisc.cmn'
#endif
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
!	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif


* check out user-specified format
	left_paren  = INDEX( string, '(' )
	right_paren = INDEX( string, ')' )
	IF ( left_paren .EQ. 0 
     .	.OR. left_paren .GE. right_paren ) GOTO 5040
	integer_form = MAX( INDEX( string, 'I' ),
     .		            INDEX( string, 'i' ) )
	IF ( integer_form .NE. 0 ) THEN
           IF ( TM_DIGIT(string(integer_form:)) )GOTO 5050
        ENDIF

* success
	status = ferr_ok
	RETURN

* syntax error
 5000	RETURN
#ifdef NO_PASSED_CONCAT	
 5040   risc_buff = string
	CALL ERRMSG( ferr_syntax, status,
     .			'unknown format or format need parenthesis: '//pCR//
     .			risc_buff, *5000 )
 5050   risc_buff = string
	CALL ERRMSG( ferr_syntax, status,
     .			'floating pt. formats only:'//risc_buff, *5000 )
#else
 5040	CALL ERRMSG( ferr_syntax, status,
     .			'unknown format or format need parenthesis: '//pCR//
     .			string, *5000 )
 5050	CALL ERRMSG( ferr_syntax, status,
     .			'floating pt. formats only:'//string, *5000 )
#endif
	END
