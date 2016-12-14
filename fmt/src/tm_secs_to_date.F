	CHARACTER*(*) FUNCTION TM_SECS_TO_DATE ( num_secs, cal_id )
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
* Will convert secs since 01-JAN-0000 00:00:00 to a date of the form
* yyyy-mm-dd hh:mm:ss
*
* Written 6/16/87 By Mark Verschell for PMEL/TMAP
*
* revision 0.00 - 06/16/87 - Initial Incarnation
* revision 0.10 - 09/19/88 - changed function to CHAR*(*) from CHAR*20 and now
*			     write to temp string to get around 20 character
*			     minimum length needed for date output
* revision 1.00 - 02/09/89 - Completely rewritten eliminating all if-tests,
*			     loops, and goto's.
* revision 2.00 - 02/20/89 - Rewritten - subtract year 0 - claculate from there
*			     year 0 special case - readded logical leap test
*			     Added some error handling (STOP's)

* V510 5/00 *sh* - trap attempt to translate a date before 0000
* V530: *acm* 12/00 add calendar types; cal_id.  This routine is from
*       Jerry's tm_calendars_secs_to_date
*       *acm* 3/01  Treat Julian calendar as a separate case.
*       *ACm* 4/24/01 INT(tot_days/366. + 307./366.)  is 0 under linux,
*                     when the expression is 1.  Use simple IF test on
*		      tot_days instead to see if we're before or after day 59.
* V692 *acm* 10/14 Trap errors in year, month, day etc before making the string 
* V693 *acm* 11/13 back off previous change. Out-of-range time info is handled
*                  elsewhere (Prev. change caused ticket 2116).
* V710  3/16 *acm* Ticket 2158: working with true monthly time axes. Pull the
*                  guts of this routine so TM_SECS_TO_YMDHMS can be called
*                  separately to return the numeric year, month, etc 

* Argument definitions
	REAL*8		num_secs
	INTEGER		cal_id

* Include files
	include 'tmap_errors.parm'
        include 'calendar.decl'
        include 'calendar.cmn'

* Local Definitions

	CHARACTER*20	tempstr
	INTEGER		year, month, day, hour, minute, second, status
	REAL		yeardays

	CALL TM_SECS_TO_YMDHMS (num_secs, cal_id, 
     .     year, month, day, hour, minute, second, status)

* trap invalid input
       IF ( num_secs .LT. 0 ) THEN
          TM_SECS_TO_DATE = '01-JAN-0000:00:00:00'
          RETURN
       ENDIF

	IF (status .NE. merr_ok) GOTO 5000

* Write the date string

C  Get calendar attributes
        CALL TM_GET_CALENDAR_ATTRIBUTES (cal_id, num_months, 
     .                      num_days, yeardays, month_names,
     .                      days_in_month, days_before_month, 
     .                      month_by_day)

	WRITE (tempstr,1000,ERR=5000) day, month_names(month), 
     .      year, hour, minute, second

1000	FORMAT(I2.2,'-',A3,'-',I4.4,' ',2(I2.2,':'),I2.2)
	TM_SECS_TO_DATE = tempstr
	GOTO 9990

* Date syntax error
 5000	CALL TM_ERRMSG (merr_erreq, status, 'TM_SECS_TO_DATE',
     .			no_descfile, no_stepfile,
     .			no_errstring, no_errstring, *9000)

 9000	STOP

* finished
9990	RETURN
	END
