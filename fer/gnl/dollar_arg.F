	SUBROUTINE DOLLAR_ARG( carg, goline, arg, lenarg, status )

* locate and return the indicated argument for this GO file

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V300: 4/21/93
* V420:  avoid breaking apart "[]" as at the comma in "var[x=1,y=2]"

* calling argument declarations:
	INTEGER	  lenarg, status
	CHARACTER carg*1, goline*(*), arg*(*)

* internal variable declarations:
	LOGICAL quote, found_comma
	INTEGER	TM_LENSTR1,lengo, iarg, goptr, start, end, karg,
     .		open_paren, open_brkt
	CHARACTER c1*1

* internal parameter declarations
	CHARACTER*1     tab
	PARAMETER     ( tab = CHAR(9) )

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
#endif

* initialize
	status = ferr_ok
	goptr = 0
	lengo = TM_LENSTR1( goline )
	found_comma = .FALSE.

* which argument has been requested?
	IF ( carg .EQ. '*' ) THEN
	   karg = 1               ! locate start of arg #1
	ELSEIF ( carg .EQ. '<' ) THEN
	   GOTO 5100     ! implement later
	ELSEIF ( carg.LT.'0' .OR. carg.GT.'9' ) THEN
	   GOTO 5100
	ELSE
	   READ ( carg, '(I1)' ) karg	   
	ENDIF

* locate the start and end of argument number karg
	DO 500 iarg = 0, karg

* skip blanks and up to one comma that precede the argument
	   DO 100 start = goptr+1, lengo
	      c1 = goline(start:start)
	      IF ( c1 .EQ. ',' ) THEN
	         IF ( found_comma ) THEN
	            end = start-1
	            goptr = start    ! (start-1 hangs on double comma)
	            GOTO 500
	         ENDIF
	         found_comma = .TRUE.
	      ELSEIF ( c1.NE.' ' .AND. c1.NE.tab ) THEN
	         GOTO 110
	      ENDIF
 100	   CONTINUE
* no more arguments in go line
	   lenarg = 0
	   arg = ' '
	   RETURN

* locate the end of the argument
* ignore commas inside of parentheses to protect "/LEV=(lo,hi,delta)"
* 10/95 ignore commas inside of sqr brackets to protect "VAR[x=1,y=2]"
 110	   quote = .FALSE.
	   open_paren = 0
	   open_brkt  = 0
	   DO 200 goptr = start, lengo
	      c1 = goline(goptr:goptr)
	      IF ( c1 .EQ. '"' ) THEN
	         quote = .NOT.quote
	      ELSEIF ( quote ) THEN
	         CONTINUE           ! skip over quoted text
	      ELSEIF ( c1 .EQ. '(' ) THEN
	         open_paren = open_paren + 1
	      ELSEIF ( c1 .EQ. ')' ) THEN
	         IF (open_paren.GT.0) open_paren = open_paren - 1
	      ELSEIF ( c1 .EQ. '[' ) THEN
	         open_brkt  = open_brkt  + 1
	      ELSEIF ( c1 .EQ. ']' ) THEN
	          IF (open_paren.GT.0) open_brkt  = open_brkt  - 1
	      ELSEIF ( c1 .EQ. ',' ) THEN
	         IF ( open_paren.EQ.0 .AND. open_brkt.EQ.0 ) THEN
	            found_comma = .TRUE.
	            end = goptr - 1
	            GOTO 500	     
	         ENDIF
	      ELSEIF (c1.EQ.' ' .OR. c1.EQ.tab ) THEN
	         end = goptr - 1
	         GOTO 500
	      ENDIF
 200	   CONTINUE
	   end = lengo     ! end of buffer is end of arg

 500	CONTINUE

* return the requested argument string
	IF ( carg .EQ. '*' .AND. end .NE. lengo ) THEN
	   end = lengo
	ELSEIF ( goline(start:start) .EQ. '"'
     .     .AND. goline(end:end)     .EQ. '"' ) THEN
* ... do not pass enclosing quotation marks
	   start = start+1
	   end = end - 1
	ENDIF
	IF ( start .GT. end ) THEN   ! beware empty quotes
              arg = ' '
              lenarg = 0
	ELSE
	   arg = goline(start:end)
	   lenarg = end - start + 1
	ENDIF

* a set of empty quotes is regarded as a place-holder for the default
	IF ( arg .EQ. ' ' ) lenarg = 0

* done
 	RETURN

* error exits
 5000	RETURN
 5100	CALL ERRMSG( ferr_syntax, status,
     .		     'illegal GO command argument: $'//carg, *5000 )
	END
