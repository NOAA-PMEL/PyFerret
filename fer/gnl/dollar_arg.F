	SUBROUTINE DOLLAR_ARG( carg, goline, arg, lenarg, status )

*
*
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
