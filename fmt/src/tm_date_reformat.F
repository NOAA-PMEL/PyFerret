	SUBROUTINE TM_DATE_REFORMAT( indate,  inform, cal_id, 
     .				     outdate, outform, do_err, status)
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
* This function will convert a date in format inform into a date in
* format outform

* *sh* 2/13/95
* *acm* needs cal_id to pass to tm_break_fmt_date
* V600 *acm* 2/06 - Fix for bug 1394, send do_err flag to tm_break_date
* v745 *acm* 12/18 Issue 1909, option to write dates in ISO8601 format
* v7441*acm* 12/18 Issue 1910: Allow dates in commands to be in ISO 8601 form, yyyy-mm-dd
*
* Arguments:
*	indate	 - input date string
*	inform	 - code for input format
*       cal_id id for calendar type of input date
*	outdate	 - output date string
*	outform	 - code for output format
*	status   - returning status
*
        LOGICAL         do_err
	CHARACTER*(*)	indate, outdate
	INTEGER		inform, outform, status, cal_id 
*
* Inclusions
*
	include 'tmap_errors.parm'
	include 'errmsg.parm'
#include "gt_lib.parm"


*
* Local definitions
*
	LOGICAL no_hms, has_hms, has_hm, has_h
	INTEGER	TM_LENSTR1, year, month, day, hour, minute, dlen, idot, noyear
	REAL	second
	CHARACTER errdate*26, endtxt*10
	
* Remove text representing fractional seconds, and store that info
	dlen = TM_LENSTR1(indate)
	idot = INDEX(indate, '.')
	IF (idot .GT. 0) THEN
	   idot = INDEX(indate, '.')
	   endtxt = indate(idot:dlen)
	   dlen = idot - 1
	ENDIF

* Break up the input date string to its various components
	CALL TM_BREAK_FMT_DATE ( inform, indate(:dlen), cal_id, year, 
     .                        month, day, hour, minute, second, 
     .                        do_err, status )
	IF ( status .NE. merr_ok ) GOTO 5000

* Keep the same resolution, e.g. if they gave 00:00:00 keep that,
* but if the input string is HH:MM don't output HH:MM:00

* The lengths below need adjusting if the incoming date has no
* year, e.g. 16-JAN "12:00:00"

	noyear = 0
	IF (inform .EQ. pdate_vax .AND. indate(7:7) .EQ. " ") noyear = 4

	no_hms  = (dlen .LE. 11-noyear)
	has_hms = (dlen .GE. 19-noyear)
	has_hm  = (.NOT.has_hms .AND. dlen.GE.16-noyear)
	has_h   = (.NOT.has_hm  .AND. .NOT.has_hms .AND. dlen.GE.13-noyear)


* Do a few sanity checks. Return a message foran incorrectly written iso-8601 date

	IF (month.LT.0 .OR. month.GT.12)  GOTO 5100
	IF (day.LT.0 .OR. day.GT.31)  GOTO 5100

	CALL TM_MAKE_FMT_DATE ( outform, outdate, year, month, day,
     .				   hour, minute, second, dlen )

	dlen = TM_LENSTR1(outdate)
	IF (idot .GT. 0) THEN
	   outdate = outdate(:TM_LENSTR1(outdate))//endtxt(:TM_LENSTR1(endtxt))
	ELSE

* Trim extra :00 's 
	   IF (no_hms) outdate = outdate(:dlen-9)
	   IF (has_h)  outdate = outdate(:dlen-6)
	   IF (has_hm) outdate = outdate(:dlen-3)
	ENDIF

* Finished
	status = merr_ok
	RETURN

* Errors
 5000	RETURN
 5100	CONTINUE
 
        IF (do_err) 
     .       CALL ERRMSG( ferr_syntax, status, 
     .             indate(:dlen)//' invalid date or not valid in this calendar ', *5000)

	END
