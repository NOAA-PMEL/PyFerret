	REAL*8 FUNCTION TM_DATE_TO_STEP ( dset_num, date, status )
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
* This function will return a REAL*8 number giving the time step for a given
* date for a particular data set. This is done by figuring the offset from
* 01-JAN-0000 00:00:00 in seconds, and then calculating the timestep.
*
* written 06/17/87 by Mark Verschell for PMEL/TMAP
*
* revision 0.00	- 06/17/87 - initial incarnation
* revision 0.01 - 07/08/88 - Returns status code, handles errors from
*			     TM_BREAK_DATE
*
* Arguments:
*	dset_num - data set number
*	date	 - date of interest
*	status   - returning status
*
	CHARACTER*20	date
	INTEGER		dset_num, status
*
* Inclusions
*
#ifdef unix
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
#endif

*
* Local definitions
*
	INTEGER		year, month, day, hour, minute, second, istat,
     .			t0year, t0mon, t0day, t0hour, t0min, t0sec
	REAL*8		date_secs, start_secs, offset_secs
	REAL*8		TM_SECS_FROM_BC

* Break up the date string to it's various components
	CALL TM_BREAK_DATE ( date, year, month, day, hour, minute, second,
     .	                     istat )
	IF ( istat .NE. merr_ok ) GOTO 9000

* Find the offset from 01-JAN-0000 00:00:00 for this date
	date_secs = TM_SECS_FROM_BC (year, month, day,
     .				     hour, minute, second)

* First find the offset from 01-JAN-0000 00:00:00 from the descriptor file
	CALL TM_BREAK_DATE (ds_t0time(dset_num), t0year, t0mon,
     .			    t0day, t0hour, t0min, t0sec, status)
	IF ( istat .NE. merr_ok ) GOTO 9100
	start_secs = TM_SECS_FROM_BC (t0year, t0mon, t0day,
     .				      t0hour, t0min, t0sec)

* Find the number of seconds since start of data set for this date
	offset_secs = date_secs - start_secs

* Divide by time step increment to get # of time step
	TM_DATE_TO_STEP = offset_secs/ds_time_unit(dset_num)

* Finished
	status = merr_ok
	GOTO 9990

* Errors
 9000	CALL TM_ERRMSG ( istat, status, 'TM_DATE_TO_STEP',
     .	                 dset_num, no_stepfile,
     .			 'DD-MMM-YYYY HH:MM:SS', date, *9990 )
 9100	CALL TM_ERRMSG ( istat, status, 'TM_DATE_TO_STEP',
     .	                 dset_num, no_stepfile,
     .			 'DD-MMM-YYYY HH:MM:SS',ds_t0time(dset_num),
     .	                 *9990 )

 9990	RETURN
	END
