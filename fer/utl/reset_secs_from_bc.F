	SUBROUTINE reset_secs_from_bc( tstep, new_tstep, 
     .                                 old_cal_id, new_cal_id, 
     .                                 status )

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
* return a REAL*8 number giving the number of seconds since 1-JAN-0000:00:00
* for tstep.  On input tstep is relative to calendar old_cal_id,
* on output tstep is relative to the calendar indicated by new_cal_id

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* revision 0.0 2/28/01

* Calling argument declarations:
	INTEGER		old_cal_id, new_cal_id, status
	REAL*8		tstep, new_tstep

* Local variable declarations:
	INTEGER		year, month, day, hour, minute, second
	REAL*8		TM_SECS_FROM_BC
	CHARACTER*20	TM_SECS_TO_DATE, date

#ifdef unix
	include 'ferret.parm'
	include 'errmsg.parm'
#endif
	IF (old_cal_id .EQ. new_cal_id ) THEN
           new_tstep = tstep
        ELSE

* Convert to date string
	   date = TM_SECS_TO_DATE(tstep, old_cal_id)
	   CALL TM_BREAK_DATE ( date, old_cal_id,
     .			  year, month, day, hour, minute, second,
     .			  status )
	   IF (status .NE. ferr_ok) GO TO 9000

* Find the offset from 01-JAN-0000 00:00:00 for this date and calendar

	   new_tstep = TM_SECS_FROM_BC (new_cal_id, year, month, day,
     .				     hour, minute, second, status)
	ENDIF

 9000	RETURN
	END
