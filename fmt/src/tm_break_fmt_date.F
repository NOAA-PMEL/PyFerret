	SUBROUTINE TM_BREAK_FMT_DATE (form, date, year, month, day,
     .					 hour, minute, second, status)
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

* ARGUMENT DEFINITIONS
*  form  - code for the date string format
*  date  - date to process
*  day, month, year, hour, minute, second - obvious
*  status - returned status
	CHARACTER*(*)	date
	INTEGER	form, day, month, year, hour, minute, second, status
*
* INCLUDE FILES
#ifdef unix
	include 'tmap_errors.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
#endif

* internal variable declarations:
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
	    CALL TM_BREAK_DATE( date, year, month, day,
     .			hour, minute, second, status)
	    RETURN

	ELSEIF ( form .EQ. pdate_numslashnum ) THEN
* ... note:  this is a quick and dirty kludge parser.  Use the udunits package
*	     for a more robust approach
* ... the format looks like (e.g.) 1992-10-8 15:15:42.5 

	   CALL TM_FTOC_STRNG( date(:dlen), dhol, slen)
	   status = TM_BREAK_FMT_DATE_C(dhol, year, month, day,
     .				        hour, minute, fsecond)
	   IF ( status .NE. 0 ) THEN
* ... last attempt -- try VAX format
	      CALL TM_BREAK_DATE( date, year, month, day,
     .			hour, minute, second, status)
	      IF ( status .NE. merr_ok ) GOTO 5100
	   ENDIF
	   second = NINT(fsecond)	! information loss ... 
	ELSE
	   STOP 'Unknown date format code'
	ENDIF

* successful translation
	status = merr_ok
	RETURN

* error exits
 5000	RETURN
 5100	CALL TM_ERRMSG (merr_badtimedef, status, 'TM_BREAL_FMT_DATE',
     .			unspecified_int4, no_stepfile,
     .			'syntax error in "'//date(:dlen),
     .			no_errstring, *5000)

	END
