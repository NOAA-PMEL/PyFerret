	SUBROUTINE TM_CALENDARS_BREAK_DATE (calendar_id,
     .     date, year, month, day, hour, minute, second, status)
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

*       J Davison 10.15.99   Copied TM_BREAK_DATE() and modified to support
*                            multiple calendars.

* ARGUMENT DEFINITIONS
*  date  - date to process
*  day, month, year, hour, minute, second - obvious
*  status - returned status
	CHARACTER*(*)	date
	INTEGER*4	day, month, year, hour, minute, second, status
*
* INCLUDE FILES
#ifdef unix
	include 'tmap_errors.parm'
#include "gt_lib.parm"
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
#endif

#ifdef unix
        include 'calendar.cmn'
#else
        INCLUDE 'TMAP_PPLV11INC:CALENDAR.CMN'
#endif

*
* LOCAL DEFINITIONS
	CHARACTER*20	fill_in
	INTEGER*4	out_state(7,22), in_state, pos, istat,
     .			leapadj, calendar_id
	INTEGER*4	STR_CASE_BLIND_COMPARE, Class

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
	fill_in = '01-'//month_names(1)//'-0000:00:00:00'
	in_state = 1
	pos = 0

* read in next char
  100	pos = pos + 1
* . Calc next state
	in_state = out_state(Class(date,pos),in_state)
* . Process next character unless this is a change of class, then output
* STATUS       1   2   3   4   5   6   7   8   9   10  11  12  13  14
	GOTO (100,100,100,100,120,100,100,100,100,130,140,100,150,100,
     .	      100,160,100,100,170,100,100,180,190,9000) in_state
* STATUS       15  16  17  18  19  20  21  22  23  24

* month found
  120	fill_in(4:6) = date(pos-2:pos)
	GOTO 100
* 1 digit day found
  130	fill_in(2:2) = date(pos-1:pos-1)
	GOTO 100
* 2 digit day found
  140	fill_in(1:2) = date(pos-2:pos-1)
	GOTO 100
* year found
  150	fill_in(8:11) = date(pos-3:pos)
	GOTO 100
* hour found
  160	fill_in(13:14) = date(pos-1:pos)
	GOTO 100
* minute found
  170	fill_in(16:17) = date(pos-1:pos)
	GOTO 100
* second found
  180	fill_in(19:20) = date(pos-1:pos)
	GOTO 100

* End of string; break into parts
  190	READ (fill_in(1:2),'(I2)') day
	READ (fill_in(8:11),'(I4)') year
	READ (fill_in(13:14),'(I2)') hour
	READ (fill_in(16:17),'(I2)') minute
	READ (fill_in(19:20),'(I2)') second
	DO 200 month = 1,num_months
	  istat = STR_CASE_BLIND_COMPARE( fill_in(4:6), 
	1	month_names(month) )
	  IF (istat .EQ. str_match) GOTO 210
  200	CONTINUE
	GOTO 9000

 210	continue

C       ***************************************
C       ******* CHECK VALIDITY OF DATES *******
C       ***************************************

******************************************************************************
* HANDLE GREGORIAN CALENDAR FIRST
******************************************************************************
        if (calendar_id .eq. gregorian) then

*          DAYS IN MONTH
	   if (month .eq. 2) then
	      if (mod(year,400) .eq. 0) then
		 leapadj = 1
	      else if (mod(year,4).eq.0 .and. mod(year,100).ne.0) then
		 leapadj = 1
	      else
		 leapadj = 0
	      endif
	   else
	      leapadj = 0
	   endif
	   if (day .gt. days_in_month(month)+leapadj
	1	.or. day .le. 0   ) goto 9000

*          HOURS IN DAY, MINUTES IN HOUR, SECONDS IN MINUTE
	   if (hour .gt. 23 .or. minute .gt. 59 .or. second .gt. 59)
	1	goto 9000


******************************************************************************
* ALL OTHER SUPPORTED AND REGULAR CALENDARS
******************************************************************************
	else if (calendar_id .eq. common   .or.
     .           calendar_id .eq. leap_yr  .or. 
     .           calendar_id .eq. days_360 .or. 
     .           calendar_id .eq. tmap_360 .or. 
     .           calendar_id .eq. peculiar) then

*          DAYS IN MONTH
	   if (day .gt. days_in_month(month) 
	1	.or. day .le. 0) goto 9000

*          HOURS IN DAY, MINUTES IN HOUR, SECONDS IN MINUTE
	   if (hour .gt. 23 .or. minute .gt. 59 .or. second .gt. 59)
	1	goto 9000
	else
           stop 'Illegal calendar in tm_calendars_break_date()'
        end if

	goto 9990

* Syntax error
 9000	status = merr_syntax
	GOTO 9999

 9990	status = merr_ok
 9999	RETURN
	END

******************************************************************************
* ROUTINE SILENCED AS IT STILL EXISTS IN ORIGINAL SOURCE FILE
******************************************************************************
CC 	INTEGER*4 FUNCTION Class ( in_str, pos )
CC *
CC * Takes a character {in_str(pos:pos)}, and determines it's class:
CC *  1 : A to Z
CC *  2 : 0 to 9
CC *  3 : blank (tab)
CC *  4 : -
CC *  5 : :
CC *  6 : other
CC *  7 : end of string
CC *
CC * ARGUMENTS
CC *  in_str - input string
CC *  pos    - character position in string
CC 	CHARACTER*(*)	in_str
CC 	INTEGER*4	pos
CC *
CC * LOCAL DEFINITIONS
CC 	CHARACTER*1	chr, blank, tab, hyphen, colon
CC 	INTEGER*4	slen
CC 	INTEGER*4	TM_LENSTR
CC 	DATA	blank	/ ' ' /
CC 	DATA	hyphen	/ '-' /
CC 	DATA	colon	/ ':' /
CC 		tab	= CHAR(9)
CC 
CC * Determine length of string
CC 	slen = TM_LENSTR( in_str )
CC * Is pos at end of string?
CC 	IF (pos .GT. slen) THEN
CC 	  class = 7
CC 	  RETURN
CC 	ENDIF
CC 
CC * Get character
CC 	chr = in_str( pos:pos )
CC * Alphabetic?
CC 	IF ( chr .GE. 'A' .AND. chr .LE. 'Z' .OR.
CC      .	     chr .GE. 'a' .AND. chr .LE. 'z'      ) THEN
CC 	  class = 1
CC * Numeric?
CC 	ELSE IF (chr .GE. '0' .AND. chr .LE. '9' ) THEN
CC 	  class = 2
CC * Blank or tab?
CC 	ELSE IF (chr .EQ. blank .OR. chr .EQ. tab ) THEN
CC 	  class = 3
CC * Hyphen?
CC 	ELSE IF (chr .EQ. hyphen) THEN
CC 	  class = 4
CC * Colon?
CC 	ELSE IF (chr .EQ. colon) THEN
CC 	  class = 5
CC * Everything else
CC 	ELSE
CC 	  class = 6
CC 	ENDIF
CC 	RETURN
CC 	END
CC 
