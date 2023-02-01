	SUBROUTINE TM_BREAK_FMT_DATE (form, date, cal_id, 
     .                                year, month, day, hour, minute, 
     .                                second, do_err, status)
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
* Will take date strings of various formats and break them into their
* various components.

* *sh* 2/13/95
* V500 *sh* 7/99 - if numslashnum reading fails, try dd-mmm-yyyy (VAX)
*		to support LDEA files with "months since 01-jan"
* V530 *acm* 12/00 - add cal_id for alternative calendar axes
* 
* 1/03 - *kob* g77 port - Need to use risc_buff for arbitrary concatenations
* V551 *acm*  2/03 - fix bug where seconds were wiped out. See cc comments
* V600 *acm* 2/06 - Fix for bug 1394, do err msg only if do_err flag true
* V67  *acm* 3/11 - If there is a T between the date and time in the unitsi 
*                   string "months since yyyy-mm-ddThh:mm:ss (as in GRIB files) 
*                   allow that. Replace T with a space. Ticket 1806.
* V72 *acm*  3/17   If there is a Z at the end, lop that off.
* v745 *acm* 12/18 Issue 1909, option to write dates in ISO8601 format
* v7441*acm* 12/18 Issue 1910: Allow dates in commands to be in ISO 8601 form, yyyy-mm-dd

* ARGUMENT DEFINITIONS
*  form  - code for the date string format
*  date  - date to process
*  cal_id  integer code for which calendar to use
*  day, month, year, hour, minute, second - obvious. Second may be a real number sec.fraction
*  status - returned status

	CHARACTER*(*)	date
	INTEGER	form, day, month, year, hour, minute, status,cal_id
	REAL	second
*
* INCLUDE FILES

	include 'tmap_errors.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'xrisc_buff.cmn'


* internal variable declarations:
        LOGICAL do_err
	INTEGER slen, isecond
	REAL	frac
	PARAMETER ( slen = 32)
	INTEGER TM_LENSTR1, TM_BREAK_FMT_DATE_C, dlen
	INTEGER*1  dhol(slen)     !c-type Hollerith string buffers

	dlen = TM_LENSTR1( date)

	IF ( form .EQ. pdate_vax ) THEN
	    CALL TM_BREAK_DATE( date, cal_id, year, month, day,
     .			hour, minute, isecond, status)
	    second = FLOAT(isecond)
	    IF (TM_LENSTR1(date) .GT. 20) THEN
	       READ (date(22:slen), *, err=5100)  frac
	       second = second + frac/100.
	    ENDIF

	    RETURN

	ELSEIF ( form .EQ. pdate_numslashnum ) THEN

* ... note:  this is a quick and dirty kludge parser.  Use the udunits package
*	     for a more robust approach
* ... the format looks like (e.g.) 1992-10-18 15:15:42.5 

*     If there is a T or a : between the date and time (as in GRIB files) replace with a space.
           IF (dlen .GT. 10) THEN
              IF (date(11:11).EQ.'T') date(11:11) = ' '
              IF (date(11:11).EQ.':') date(11:11) = ' '
           ENDIF

* If there is a Z at the end, lop that off.

           IF (date(dlen:dlen) .EQ. 'Z') date(dlen:dlen) = ' '

	   CALL TM_FTOC_STRNG( date(:dlen), dhol, slen)
	   status = TM_BREAK_FMT_DATE_C(dhol, year, month,
     .				        day, hour, minute, second)

	   IF ( status .NE. 0 ) THEN
* ... last attempt -- try VAX format
	      CALL TM_BREAK_DATE( date, cal_id, year, month, day,
     .			hour, minute, isecond, status)
	      IF ( status .NE. merr_ok ) GOTO 5200
	      
	      second = FLOAT(isecond)
	      IF (TM_LENSTR1(date) .GT. 20) THEN
	          READ (date(22:slen), *, err=5100)  frac
	          second = second + frac/100.
	       ENDIF

	   ENDIF

	ELSE
	   STOP 'Unknown date format code'
	ENDIF

* successful translation
	status = merr_ok
	RETURN

* error exits
 5000	RETURN
 5100	risc_buff = date(:dlen)
        IF (do_err) THEN
          CALL TM_ERRMSG (merr_badtimedef, status, 'TM_BREAK_FMT_DATE',
     .			unspecified_int4, no_stepfile,
     .			'syntax error in "'//risc_buff,
     .			no_errstring, *5000)
        ELSE
           dlen = TM_LENSTR1(risc_buff)
           CALL WARN (
     .       'cannot parse "units since date", date= "'//
     .        risc_buff(:dlen)//'"'//'; leaving units string')
           status = merr_badtimedef
        ENDIF
	GOTO 5000
          
 5200	risc_buff = date(:dlen)
        IF (do_err) THEN
        CALL TM_ERRMSG (merr_badtimedef, status, 'TM_BREAK_FMT_DATE',
     .			unspecified_int4, no_stepfile,
     .			date(:dlen)//' is not valid in this calendar ',
     .			no_errstring, *5000)
        ELSE
           dlen = TM_LENSTR1(risc_buff)
           CALL WARN (
     .       'cannot parse "units since date", date= "'//
     .        risc_buff(:dlen)//'"'//'; leaving units string')
           status = merr_badtimedef
        ENDIF
	GOTO 5000
	END

