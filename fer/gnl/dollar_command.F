	SUBROUTINE DOLLAR_COMMAND( cmnd,lencmnd,goline,did_sub,status )

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
* parse the command line replacing "dollar" arguments in GO files
* with the appropriate strings

* formats supported:
*	$n 		- argument n (1 through max)
*	$0		- name of Go file
*	$*		- all arguments, 1 through max
*       ($nn)	        - to allow 2-digit argument numbers
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
* V510: 3/00 *sh* bug fix - in the syntax "$1" the final " caused iptr mis-set
* V531: 4/01 *sh* - allow 2-digit argument numbers
*		  - increased argument buffers 255 -> 511
* 2/03 *kob* - g77 port - g77 won't allow intrinsic functions in PARAMETER
*                         statements.  use character constant instead
*            - Need to use risc_buff for arbitrary concatenations
*       *acm* 3/12 cleanup ifdefs and unnecessary include files

        IMPLICIT NONE
* calling argument declarations:
	LOGICAL   did_sub
	INTEGER	  lencmnd, status
	CHARACTER cmnd*(*), goline*(*)

* internal variable declarations:
* d default string
* e error string
* o options string
* r replacement strings

	LOGICAL coaching, in_squote, has_parens
	INTEGER	TM_LENSTR1,
     .		lenbuff, iptr, dollar, lenarg, nexchar, idig
        CHARACTER arg*511, passed_arg*511, c1*1, carg*2

* local parameter declarations
        CHARACTER*1     sing_quote
        CHARACTER*1	tab
	PARAMETER     ( tab = CHAR(9))
        PARAMETER     ( sing_quote = CHAR(39) )

	include 'ferret.parm'
	include 'errmsg.parm'
	include 'gfdl_vms.parm'
	include 'xrisc.cmn'

* initialize
	nexchar = 1
	did_sub = .FALSE.
	status = ferr_ok
        lenbuff = LEN( cmnd )
	in_squote = .FALSE.

* init for next dollar arg
 10	coaching = .FALSE.
	in_squote = .FALSE.

* locate the next dollar sign  (ignore "$" as last character in command)
	DO 50 dollar = nexchar, lencmnd-1
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

* note: multiple digit argument number must be enclosed in parentheses.
* this because $15 might legitimately be "$1" prepended to "5"

* is this a paren-surrounded argument number?
* Note: the parsing actually gets done twice ... this is an add-on hack!
* check syntax: ($nn%stuff%)
	has_parens = .FALSE.
	IF (dollar .GT. 1) THEN
	   IF (cmnd(dollar-1:dollar-1) .NE. '(') GOTO 65
	   IF (carg.LT.'0' .OR. carg.GT.'9') GOTO 65
	   iptr = dollar+2
	   IF (iptr .GT. lencmnd) GOTO 65
	   c1 = cmnd(iptr:iptr)
	   IF (c1.GE.'0' .AND. c1.LE.'9') iptr = iptr + 1
	   IF (iptr .GT. lencmnd) GOTO 65
	   c1 = cmnd(iptr:iptr)
           coaching = c1.GE.'!' .AND. c1.LE.'&'
	   IF (coaching) THEN
             DO 62 iptr = iptr+1, lencmnd
                 IF ( cmnd(iptr:iptr) .EQ. c1 ) GOTO 64
 62	      CONTINUE
              GOTO 65       ! unterminated option string - red herring?
 64	      iptr = iptr + 1
	      IF (iptr .GT. lencmnd) GOTO 65
	      c1 = cmnd(iptr:iptr)
	   ENDIF
	   IF (c1 .NE. ')') GOTO 65
	   has_parens = .TRUE.
	ENDIF
 65	CONTINUE

* find a second digit, if any
	nexchar = dollar + 2
	IF (carg.GE.'0' .AND. carg.LE.'9'
     .	.AND. lencmnd .GT. dollar+1 ) THEN
	   DO 70 iptr = dollar+2, lencmnd
	     IF (cmnd(iptr:iptr).LT.'0'.OR.cmnd(iptr:iptr).GT.'9') GOTO 72
 70	   CONTINUE
 72	   carg = ' '
	   idig = iptr-dollar-1
	   IF ( idig.GT.2 ) GOTO 5150
	   carg(dollar-iptr+4:2) = cmnd(dollar+1:iptr-1) ! right justify
	   nexchar = iptr  ! next character after digits (changes later)
	ELSE
	   idig = 1
	   c1 = carg(1:1)
	   carg = ' '
	   carg(2:2) = c1  ! right justify
	ENDIF

	IF (has_parens) THEN
	   dollar = dollar - 1 ! text to replace starts at "("
	ENDIF

* is there a coaching string following  "$x" ?  (e.g. $1"default string")
        coaching = nexchar+1 .LE. lencmnd
        IF ( coaching ) THEN
           c1 = cmnd(nexchar:nexchar)
           coaching = c1.GE.'!' .AND. c1.LE.'&'
        ENDIF

* locate the matching terminator of the coaching string
        IF ( coaching ) THEN
           DO 100 iptr = nexchar+1, lencmnd
              IF ( cmnd(iptr:iptr) .EQ. c1 ) GOTO 200
 100       CONTINUE
           coaching = .FALSE.       ! unterminated option string - red herring?
        ENDIF

* get the argument given on the command line
 200    CALL DOLLAR_ARG( carg, goline, arg, lenarg, status )
        IF ( status .NE. ferr_ok ) RETURN
	passed_arg = arg	! a copy used for fancy editing 3/2

* modify the argument if coaching was provided
	IF ( coaching ) THEN
	   IF (nexchar+1 .GT. iptr-1) GOTO 5400
	   CALL COACHED_STR_SUB('argument $'//carg(3-idig:), passed_arg,
     .		cmnd(nexchar+1:iptr-1), arg, lenarg, status ) 
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   nexchar = iptr+1   ! after termination of coaching
	ELSEIF ( arg .EQ. " " ) THEN
	   GOTO 5500
	ENDIF

* move past closinging paren
	IF (has_parens) THEN
	   DO 210 iptr = nexchar, lencmnd
	      IF ( cmnd(iptr:iptr) .EQ. ')' ) THEN
	        nexchar = iptr + 1
	        GOTO 300
	      ELSEIF ( cmnd(iptr:iptr) .NE. ' '
     .	       .AND.   cmnd(iptr:iptr) .NE. tab ) THEN
	        GOTO 5300   ! should never arrive here
	      ENDIF
 210	   CONTINUE
	   GOTO 5300
	ENDIF

* replace "$n..." with argument text or coached alternative
 300	IF ( (dollar-1)+lenarg+(lencmnd-nexchar-1) .GT. lenbuff ) THEN
           GOTO 5100
        ELSE
	   iptr = lencmnd ! temporary buffer
	   risc_buff = cmnd(nexchar:)
           cmnd(dollar:) = arg(1:lenarg) // risc_buff
           lencmnd = (dollar-1) + lenarg + (lencmnd-nexchar+1)   ! new length
	   nexchar = nexchar + (lencmnd-iptr) ! shift by increase of cmnd len
        ENDIF

* go back for the next argument
        did_sub = .TRUE.
        GOTO 10

* error exits
 5000   RETURN
 5100   CALL ERRMSG( ferr_invalid_command, status,
     .       'command argument strings are too long', *5000)
 5150	risc_buff = cmnd(dollar:iptr)
	CALL ERRMSG( ferr_syntax, status,
     .       'arg number not in 1-99 range: '//risc_buff, *5000 )
 5300   risc_buff = cmnd(dollar:iptr)
	CALL ERRMSG( ferr_syntax, status,
     .       'missing closing paren: '//risc_buff, *5000 )
 5400   risc_buff = cmnd(dollar:iptr)
	CALL ERRMSG( ferr_syntax, status,
     .       'empty argument replacement string: '//risc_buff,
     .		 *5000 )
 5500   risc_buff = goline
	CALL ERRMSG( ferr_invalid_command, status,
     .       'Value needed for argument '//carg(3-idig:)//' in command'//
     .        pCR//'GO '//risc_buff(:TM_LENSTR1(goline)), *5000)

	END


