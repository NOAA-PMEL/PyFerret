	SUBROUTINE DOLLAR_COMMAND( cmnd,lencmnd,goline,did_sub,status )

* parse the command line replacing "dollar" arguments in GO files
* with the appropriate strings

* formats supported:
*	$n 		- argument n (1 through max)
*	$0		- name of Go file
*	$*		- all arguments, 1 through max
* these may be followed by the coached string substitution options
* described in coached_str_sub.F

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V300: 4/21/93
*       5/28/93 - ignore $'s that aren't followed by digits or special chars
* V301: 2/15/94 - allow "*" as a choice that matches all args
* V320 8/94 *kob* - IBM port - had to use risc buff for string concats
*      11/94 *sh* - IBM port - one more application of risc buff at line 500
*     3/2/95 *sh* - allow "*" to appear in the RIGHT hand side of the
*		    substitution string
*     3/24/95  - fix bug introduced 3/2 
* V420	2/29/96	- detach the "coaching" code into a separate routine so it
*		can be used in symbol substitution as well

* calling argument declarations:
	LOGICAL   did_sub
	INTEGER	  lencmnd, status
	CHARACTER cmnd*(*), goline*(*)

* internal variable declarations:
* d default string
* e error string
* o options string
* r replacement strings

	LOGICAL coaching, in_squote
	INTEGER	TM_LENSTR1,
     .		lenbuff, iptr, dollar, lenarg
        CHARACTER*255 arg*255, passed_arg*255, c1*1, carg*1

* local parameter declarations
        CHARACTER*1     sing_quote
        PARAMETER     ( sing_quote = CHAR(39) )

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/gfdl_vms.parm'
#ifdef NO_PASSED_CONCAT
	include 'ferret_cmn/xrisc.cmn'
#endif
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:GFDL_VMS.PARM'
#endif

* initialize
	iptr = 0
	did_sub = .FALSE.
	status = ferr_ok
        lenbuff = LEN( cmnd )
	in_squote = .FALSE.

* init for next dollar arg
 10	coaching = .FALSE.
	in_squote = .FALSE.

* locate the next dollar sign  (ignore "$" as last character in command)
	DO 50 dollar = iptr+1, lencmnd-1
	   c1 = cmnd(dollar:dollar)
	   IF ( c1 .EQ. sing_quote ) THEN
	      in_squote = .NOT.in_squote
	   ELSEIF ( in_squote ) THEN
	      CONTINUE                     ! to protect PLOT+ 'PPL$WIDTH', etc.
	   ELSEIF ( c1 .EQ. '$' ) THEN     ! 5/93 ignore if wrong carg
	      carg = cmnd(dollar+1:dollar+1)
	      IF ( carg .EQ. '*'
     .        .OR. carg .EQ. '<'
     .	      .OR. (carg.GE.'0' .AND. carg.LE.'9') ) GOTO 60
	   ENDIF
 50	CONTINUE

* done - no more dollar signs (could check in_squote for syntax error ...)
	RETURN

* found a significant "$"
 60	CONTINUE

* is there a coaching string following  "$x" ?  (e.g. $1"default string")
        coaching = dollar+3 .LE. lencmnd
        IF ( coaching ) THEN
           c1 = cmnd(dollar+2:dollar+2)
           coaching = c1.GE.'!' .AND. c1.LE.'&'
        ENDIF

* locate the matching terminator of the coaching string
        IF ( coaching ) THEN
           DO 100 iptr = dollar+3, lencmnd
              IF ( cmnd(iptr:iptr) .EQ. c1 ) GOTO 200
 100       CONTINUE
           coaching = .FALSE.       ! unterminated option string - red herring?
        ELSE
           iptr = dollar + 1
        ENDIF

* get the argument given on the command line
 200    CALL DOLLAR_ARG( carg, goline, arg, lenarg, status )
        IF ( status .NE. ferr_ok ) RETURN
	passed_arg = arg	! a copy used for fancy editing 3/2

* modify the argument if coaching was provided
	IF ( coaching ) THEN
	   IF (dollar+3 .GT. iptr-1) GOTO 5400
	   CALL COACHED_STR_SUB('argument $'//carg, passed_arg,
     .		cmnd(dollar+3:iptr-1), arg, lenarg, status ) 
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ELSEIF ( arg .EQ. " " ) THEN
	   GOTO 5500
	ENDIF

* replace "$n..." with argument text or coached alternative
 500    IF ( (dollar-1)+lenarg+(lencmnd-iptr) .GT. lenbuff ) THEN
           GOTO 5100
        ELSE
#ifdef AIX_XLF
	   risc_buff = cmnd(iptr+1:)
           cmnd(dollar:) = arg(1:lenarg) // risc_buff
           lencmnd = (dollar-1) + lenarg + (lencmnd-iptr)   ! new length
#else
           cmnd(dollar:) = arg(1:lenarg) // cmnd(iptr+1:)
           lencmnd = (dollar-1) + lenarg + (lencmnd-iptr)   ! new length
#endif
        ENDIF

* go back for the next argument
        did_sub = .TRUE.
        GOTO 10

* error exits
 5000   RETURN
 5100   CALL ERRMSG( ferr_invalid_command, status,
     .       'command argument strings are too long', *5000)
 5400   CALL ERRMSG( ferr_syntax, status,
     .       'empty argument replacement string: '//cmnd(dollar:iptr),
     .		 *5000 )
 5500   CALL ERRMSG( ferr_invalid_command, status,
     .       'Value needed for argument '//carg//' in command'//
     .        pCR//'GO '//goline(:TM_LENSTR1(goline)), *5000)
	END
