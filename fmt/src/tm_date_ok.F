	LOGICAL FUNCTION TM_DATE_OK( date, cal_id )

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
* determine whether a date string is acceptably formatted

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* 7/92 *sh* 
* V530: *acm* 12/00 alternative calendar types; cal_id
* v7441*acm* 12/18 Issue 1910: Allow dates in commands to be in ISO 8601 form, yyyy-mm-dd
* v751 *acm*  6/19 Issue 1940: Require two dashes before trying to parse as an ISO 8601
* v76  1/2020 *acm* Getting error-reporting correct, logical time_err set when an 
*                  error was reported  translating iso-times and time invalid on 
*                  the given calendar: no need for more messages, e.g. translate_limit

     include 'tmap_errors.parm'
#include "gt_lib.parm"
	include 'calendar.decl'
	include 'calendar.cmn'

* calling argument declarations:
      CHARACTER*(*) date
      INTEGER cal_id

* internal variable declarations:
      INTEGER	TM_LENSTR1, i, ib, yr, mth, day, hr, min, sec, 
     .		slen, dash1, dash2, status
      REAL*8	value
      CHARACTER c*1, buff*26, buff2*26

* local parameter declarations:
      CHARACTER*1     tab
      PARAMETER     ( tab = CHAR(9))

	TM_DATE_OK = .FALSE.

* First, is it a date at all?  At least, it must start with a digit
* and contain a colon or a dash.  (some of the tests from TM_DATE)

* skip leading blanks
	slen = TM_LENSTR1(date)
	ib = 0
	DO i = 1, slen
	   c = date(i:i)
	   IF ( c .EQ. ' ' .OR. c .EQ. tab ) THEN
	      ib = i
	   ELSE
	      GOTO 200
	   ENDIF
	ENDDO
* buff is blank
	RETURN

* is first character a digit ?
 200    IF ( c .LT. '0' .OR. c .GT. '9' ) RETURN

	buff = date
	IF (ib.GT.0) buff = date(ib+1:slen)
	slen = TM_LENSTR1(buff)

* is there a colon or a dash?
      IF ( INDEX( buff, ':') .EQ. 0 .AND. INDEX( buff, '-') .EQ. 0)  RETURN

* Try translating the date (without any fractional seconds).
      slen = TM_LENSTR1(buff)
      IF (INDEX(buff, '.') .GT. 0) slen = INDEX(buff, '.') - 1

      CALL TM_BREAK_DATE( buff(:slen), cal_id,
     .                   yr, mth, day, hr, min, sec, status )

c 300  CONTINUE

* If couldn't parse the date, is it an ISO8601 formatted date? Is so translate 
* the date string to default date format.

      IF (status .NE. merr_ok) THEN
      
* Require 2 dashes to translate as an ISO8601 date
* Not starting with a dash, but at least with yy-

	 TM_DATE_OK = .FALSE.
	 dash1 = INDEX(buff, '-') 
	 IF (dash1 .LT. 3) GOTO 5000 
	 buff2 = buff(dash1+1:slen)
	 dash2 = INDEX(buff(dash1+1:slen), '-')
	 IF (dash2 .EQ. 0) GOTO 5000
         
	 CALL TM_DATE_REFORMAT( buff, pdate_numslashnum, cal_id,
     .                          buff2, pdate_vax, .TRUE., status)
	 IF ( status .EQ. merr_ok ) date = buff2
         IF (status .NE. merr_ok) time_err = .TRUE.

      ENDIF

      TM_DATE_OK = status .EQ. merr_ok


 5000 RETURN
      END
