	SUBROUTINE CD_RD_YMD_1(cdfid,t1id,t0,tunit,index,dval,status)

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
* read a single time value from a "yymmddhhmmss" time axis
* if T0 is not yet known determine an appropriate value
* return time value as a double precision time relative to t0

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/95  -- specifically for accessing CDC files
* V530 *acm* 1/01 add calendar ID in TM_BREAK_DATE, TM_SECS_FROM_BC
*	    	  calendar is assumed Gregorian
* V65  *acm* 1/10  Change all netcdf2-style calls and parameter names to 
*                  netcdf3-style calls, which are also consistent with netcdf4.

* argument definitions:
*	cdfid	 - netCDF id number for already opened CDF file
*       t1id     - netCDF variable id of "time"
*       t0       - T0 string for the time axis
*       tunit    - number of seconds per axis time unit
*       index    - subscript of time axis to return
*       dval     - double precision value returned
*       status   - TMAP library return status

* argument declarations
      INTEGER cdfid, t1id, index, status
      REAL tunit
      REAL*8  dval
      CHARACTER*(*) t0

      include 'netcdf.decl'
#include "netcdf.inc"
      include 'tmap_errors.parm'

* internal variable declarations:
      INTEGER cdfstat, mon, day, yr, hour, min, sec,
     .        t0year, t0mon, t0day, t0hour, t0min, t0sec

      INTEGER cal_id
      REAL*8  TM_SECS_FROM_BC
      REAL*8  yymmdd, date_secs, start_secs, offset_secs

* internal parameter declarations
      REAL*8      p1900, p0010
      PARAMETER ( p1900 = 693961. * 24 * 60 * 60,
     .		  p0010 = 3653.   * 24 * 60 * 60 )

* read the yymmdd value
      cdfstat = NF_GET_VAR1_DOUBLE(cdfid, t1id, index, yymmdd)

      IF ( cdfstat .NE. NF_NOERR ) CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_RD_YMD_1', cdfid, t1id,
     .       'yymmdd time', no_errstring, *5000 )

* convert the 2 EPIC integers to a date
      CALL TM_YMD_TIME_CONVRT( yymmdd,
     .                         mon, day, yr, hour, min, sec)

* determine seconds from BC. 
* For this routine set the calendar ID to gregorian always...

      cal_id = 1
      date_secs  = TM_SECS_FROM_BC (cal_id, yr  , mon, day,
     .                              hour, min, sec, status)
      IF (status .ne. merr_ok) GO TO 5000

* determine the T0 value to use
      IF ( t0 .EQ. ' ' ) THEN 
         IF ( date_secs .GT. p1900 ) THEN
            t0 = '1-JAN-1900'
         ELSEIF ( date_secs .LT. p0010 ) THEN
            t0 = '1-JAN-0000'
         ELSE
	    t0 = '1-JAN-1700'
         ENDIF
      ENDIF

* break apart the T0 offset date 
      CALL TM_BREAK_DATE (t0, cal_id, t0year, t0mon,
     .                    t0day, t0hour, t0min, t0sec, status)
      IF (status .ne. merr_ok) GO TO 5000

      start_secs = TM_SECS_FROM_BC (cal_id, t0year, t0mon, t0day,
     .                              t0hour, t0min, t0sec, status)
      IF (status .ne. merr_ok) GO TO 5000

* Find the number of seconds since T0
      offset_secs = date_secs - start_secs

* Divide by time step increment to get # of time step
      dval = offset_secs/tunit

      status = merr_ok
      RETURN

* error exit
 5000 RETURN

      END


