	SUBROUTINE TM_BREAK_DATE (date, cal_id, year, month, day,
     .				  hour, minute, second, status)
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
*
* Will take date string composed of DD-MMM-YYYY:HH:MM:SS with optional format:
* DD-MMM:HH:MM:SS and break it up to it's various components.
*
* written 06/17/87 by Mark Verschell for PMEL/TMAP
*
* revision 0.00 - 06/17/87 - Initial Incarnation
* revision 1.00 - 07/07/88 - More conclusive checking of date, bad dates are
*			     trapped and flagged as errors
* revision 1.01 - 01/13/89 - Allows : between day and time parts now
* revision 1.10 - 02/08/89 - Allows year to be left off - defaults to 0000
* revision 1.11 - 02/13/89 - Allows multiple blanks between date and hour
* revision 1.12 - 02/20/89 - Allows dates of form "dd-mmm" - defaults to 0000
* 3/22/95 - check for non-positive DAY value as an error
* V530: *acm* 12/00 alternative calendar types; cal_id
* V63  *acm* 10/09 Changes for gfortran build
* V6.74 3/12 *acm* cleanup ifdefs and unnecessary include files
* 5/2020 Issue 1906: treat Proleptic-Gregorian identically to Gregorian
* 1/2023 *acm* Code cleanup: Replace the computed GOTO statement 
*                  "GOTO ( GOTO (100,100,, ... ) in_state"  with IF blocks

* ARGUMENT DEFINITIONS
*  date  - date to process
*  day, month, year, hour, minute, second - obvious
*  status - returned status
	CHARACTER*(*)	date
	INTEGER*4	day, month, year, hour, minute, second, status,
     .                  cal_id
*
* INCLUDE FILES
	include 'tmap_errors.parm'
        include 'calendar.decl'
        include 'calendar.cmn'
#include "gt_lib.parm"
*
* LOCAL DEFINITIONS
	CHARACTER*20	fill_in
	INTEGER*4	out_state(7,22), in_state, pos, istat, leapadj
	INTEGER*4	STR_CASE_BLIND_COMPARE, Class
	REAL		yeardays

* Table gives output state based on input state (right hand column)
*	A = alphabetic	N = numeric	b = blank	- = minus sign
*	: = colon	* = asterik	E = end of date
*		            A   N   b   -   :   *   E
	DATA	out_state / 02, 03, 01, 24, 24, 24, 24,			! 1
     .			    04, 24, 24, 24, 24, 24, 24,			! 2
     .			    24, 09, 24, 10, 24, 24, 24,			! 3
     .			    05, 24, 24, 24, 24, 24, 24,			! 4
     .			    24, 24, 14, 06, 14, 24, 23,			! 5
     .			    24, 07, 24, 24, 24, 24, 24,			! 6
     .			    24, 08, 24, 24, 24, 24, 24,			! 7
     .			    24, 12, 24, 24, 24, 24, 24,			! 8
     .			    24, 12, 24, 11, 24, 24, 24,			! 9
     .			    02, 24, 24, 24, 24, 24, 24,			!10
     .			    02, 24, 24, 24, 24, 24, 24,			!11
     .			    24, 13, 24, 24, 24, 24, 24,			!12
     .			    24, 24, 14, 24, 14, 24, 23,			!13
     .			    24, 15, 14, 24, 24, 24, 24,			!14
     .			    24, 16, 24, 24, 24, 24, 24,			!15
     .			    24, 24, 24, 24, 17, 24, 23,			!16
     .			    24, 18, 24, 24, 24, 24, 24,			!17
     .			    24, 19, 24, 24, 24, 24, 24,			!18
     .			    24, 24, 24, 24, 20, 24, 23,			!19
     .			    24, 21, 24, 24, 24, 24, 24,			!20
     .			    24, 22, 24, 24, 24, 24, 24,			!21
     .			    24, 24, 24, 24, 24, 24, 23/			!22

* initialize
	day = 0
	month = 0
	year = 0
	hour = 0
	minute = 0
	second = 0
	fill_in = '01-JAN-0000:00:00:00'
	in_state = 1
	pos = 0

* Look at next char
  100	pos = pos + 1

* . Calc next state
	in_state = out_state(Class(date,pos),in_state)

* . Process next character unless this is a change of class, THEN output

      IF (in_state .EQ.  1 .OR.
     .    in_state .EQ.  2 .OR.
     .    in_state .EQ.  3 .OR.
     .    in_state .EQ.  4 .OR.
     .    in_state .EQ.  6 .OR.
     .    in_state .EQ.  7 .OR.
     .    in_state .EQ.  8 .OR.
     .    in_state .EQ.  9 .OR.
     .    in_state .EQ. 12 .OR.
     .    in_state .EQ. 14 .OR.
     .    in_state .EQ. 15 .OR.
     .    in_state .EQ. 17 .OR.
     .    in_state .EQ. 18 .OR.
     .    in_state .EQ. 20 .OR.
     .    in_state .EQ. 21) GOTO 100  ! get next character
	
* month found
      IF (in_state .EQ. 5) THEN
	fill_in(4:6) = date(pos-2:pos)
	GOTO 100
      ENDIF

* 1 digit day found
      IF (in_state .EQ. 10) THEN
	fill_in(2:2) = date(pos-1:pos-1)
	GOTO 100
      ENDIF

* 2 digit day found
      IF (in_state .EQ. 11) THEN
	fill_in(1:2) = date(pos-2:pos-1)
	GOTO 100
      ENDIF

* year found
      IF (in_state .EQ. 13) THEN
	fill_in(8:11) = date(pos-3:pos)
	GOTO 100
      ENDIF

* hour found
      IF (in_state .EQ. 16) THEN
	fill_in(13:14) = date(pos-1:pos)
	GOTO 100
      ENDIF

* minute found
      IF (in_state .EQ. 19) THEN
	fill_in(16:17) = date(pos-1:pos)
	GOTO 100
      ENDIF

* second found
      IF (in_state .EQ. 22) THEN
	fill_in(19:20) = date(pos-1:pos)
	GOTO 100
      ENDIF

* End of string; break into parts
      IF (in_state .EQ. 23) THEN
	READ (fill_in(1:2),'(I2)') day
	READ (fill_in(8:11),'(I4)') year
	READ (fill_in(13:14),'(I2)') hour
	READ (fill_in(16:17),'(I2)') minute
	READ (fill_in(19:20),'(I2)') second
      ENDIF

C  Get calendar attributes

      CALL TM_GET_CALENDAR_ATTRIBUTES (cal_id, num_months, 
     .                    num_days, yeardays, month_names, 
     .                    days_in_month, days_before_month,
     .                    month_by_day)
	DO month = 1,12
	  istat = STR_CASE_BLIND_COMPARE( fill_in(4:6), 
     .       month_names(month) )
	  IF (istat .EQ. str_match) GOTO 210
	ENDDO
	GOTO 9000

 210	continue

C       ***************************************
C       ******* CHECK VALIDITY OF DATES *******
C       ***************************************


******************************************************************************
* HANDLE GREGORIAN CALENDAR FIRST
******************************************************************************
        IF (cal_id.eq.gregorian .or. cal_id.eq.proleptic) THEN

*          DAYS IN MONTH
	   IF (month .EQ. 2) THEN
	      IF (mod(year,400) .EQ. 0) THEN
		 leapadj = 1
	      ELSE IF (mod(year,4).EQ.0 .AND. mod(year,100).NE.0) THEN
		 leapadj = 1
	      ELSE
		 leapadj = 0
	      ENDIF
	   ELSE
	      leapadj = 0
	   ENDIF
	   IF (day .GT. days_in_month(month)+leapadj
     .         .OR. day .le. 0   ) GOTO 9000

*          HOURS IN DAY, MINUTES IN HOUR, SECONDS IN MINUTE
	   IF (hour .GT. 23 .OR. minute .GT. 59 .OR. second .GT. 59)
     .         GOTO 9000




******************************************************************************
* HANDLE JULIAN CALENDAR 
******************************************************************************
        ELSE IF (cal_id .EQ. julian) THEN

*          DAYS IN MONTH
	   IF (month .EQ. 2) THEN
	      IF (mod(year,4) .EQ. 0) THEN
		 leapadj = 1
	      ELSE
		 leapadj = 0
	      ENDIF
	   ELSE
	      leapadj = 0
	   ENDIF
	   IF (day .GT. days_in_month(month)+leapadj
     .         .OR. day .le. 0   ) GOTO 9000

*          HOURS IN DAY, MINUTES IN HOUR, SECONDS IN MINUTE
	   IF (hour .GT. 23 .OR. minute .GT. 59 .OR. second .GT. 59)
     .         GOTO 9000


******************************************************************************
* ALL OTHER SUPPORTED AND REGULAR CALENDARS
******************************************************************************

	ELSE IF (cal_id .GT. gregorian   .AND.  cal_id .NE. julian .AND.
     .           cal_id .LE. max_calendars) THEN


*          DAYS IN MONTH
	   IF (day .GT. days_in_month(month) 
     .         .OR. day .le. 0) GOTO 9000

*          HOURS IN DAY, MINUTES IN HOUR, SECONDS IN MINUTE
	   IF (hour .GT. 23 .OR. minute .GT. 59 .OR. second .GT. 59)
     .         GOTO 9000
	ELSE
           status = merr_syntax
	   CALL ERRMSG (status, status, 
     .       'Illegal calendar in tm_break_date()', *9999)
        END IF

	GOTO 9990

* Syntax error

 9000	status = merr_syntax
	GOTO 9999

 9990	status = merr_ok

 9999	RETURN
	END

	INTEGER*4 FUNCTION Class ( in_str, pos )
*
* Takes a character {in_str(pos:pos)}, and determines it's class:
*  1 : A to Z
*  2 : 0 to 9
*  3 : blank (tab)
*  4 : -
*  5 : :
*  6 : other
*  7 : end of string
*
* ARGUMENTS
*  in_str - input string
*  pos    - character position in string
	CHARACTER*(*)	in_str
	INTEGER*4	pos
*
* LOCAL DEFINITIONS
	CHARACTER*1	chr, blank, tab, hyphen, colon
	INTEGER*4	slen
	INTEGER*4	TM_LENSTR
	DATA	blank	/ ' ' /
	DATA	hyphen	/ '-' /
	DATA	colon	/ ':' /
		tab	= CHAR(9)

* Determine length of string
	slen = TM_LENSTR( in_str )
* Is pos at end of string?
	IF (pos .GT. slen) THEN
	  class = 7
	  RETURN
	ENDIF

* Get character
	chr = in_str( pos:pos )
* Alphabetic?
	IF ( chr .GE. 'A' .AND. chr .LE. 'Z' .OR.
     .	     chr .GE. 'a' .AND. chr .LE. 'z'      ) THEN
	  class = 1
* Numeric?
	ELSE IF (chr .GE. '0' .AND. chr .LE. '9' ) THEN
	  class = 2
* Blank or tab?
	ELSE IF (chr .EQ. blank .OR. chr .EQ. tab ) THEN
	  class = 3
* Hyphen?
	ELSE IF (chr .EQ. hyphen) THEN
	  class = 4
* Colon?
	ELSE IF (chr .EQ. colon) THEN
	  class = 5
* Everything ELSE
	ELSE
	  class = 6
	ENDIF
	RETURN
	END
