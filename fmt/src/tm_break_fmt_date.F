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


* ARGUMENT DEFINITIONS
*  form  - code for the date string format
*  date  - date to process
*  cal_id  integer code for which calendar to use
*  day, month, year, hour, minute, second - obvious
*  status - returned status
	CHARACTER*(*)	date
	INTEGER	form, day, month, year, hour, minute, second, status,
     .          cal_id
*
* INCLUDE FILES

	include 'tmap_errors.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'xrisc_buff.cmn'


* internal variable declarations:
        LOGICAL do_err
	INTEGER slen
	PARAMETER ( slen = 32)
	INTEGER TM_LENSTR1, TM_BREAK_FMT_DATE_C, dlen
	REAL fsecond
#ifdef sun
	BYTE       dhol(slen)     !c-type Hollerith string buffers
#else
	INTEGER*1  dhol(slen)     !c-type Hollerith string buffers
#endif

	dlen = TM_LENSTR1( date)

	IF ( form .EQ. pdate_vax ) THEN
	    CALL TM_BREAK_DATE( date, cal_id, year, month, day,
     .			hour, minute, second, status)
	    RETURN

	ELSEIF ( form .EQ. pdate_numslashnum ) THEN

* ... note:  this is a quick and dirty kludge parser.  Use the udunits package
*	     for a more robust approach
* ... the format looks like (e.g.) 1992-10-8 15:15:42.5 

	   CALL TM_FTOC_STRNG( date(:dlen), dhol, slen)
	   status = TM_BREAK_FMT_DATE_C(dhol, year, month,
     .				        day, hour, minute, fsecond)
	   second = NINT(fsecond)	! information loss ...

	   IF ( status .NE. 0 ) THEN
* ... last attempt -- try VAX format
	      CALL TM_BREAK_DATE( date, cal_id, year, month, day,
     .			hour, minute, second, status)
	      IF ( status .NE. merr_ok ) GOTO 5100
	   ENDIF

cc   Move this up after TM_BREAK_FMT_DATE_C -- if tm_break_fmt_date_c did 
cc   not set fsecond, don't want to wipe out the seconds that were correctly
cc   set by TM_BREAK_DATE

cc	   second = NINT(fsecond)	! information loss ... 

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
          

	END

