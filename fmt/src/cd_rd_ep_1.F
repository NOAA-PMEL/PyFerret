	SUBROUTINE CD_RD_EP_1(cdfid,t1id,t2id,t0,tunit,index,dval,status)

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
* read a single time value from a 2-integer EPIC time axis
* and return it as a double precision time relative to t0

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92

* argument definitions:
*	cdfid	 - netCDF id number for already opened CDF file
*       t1id     - netCDF variable id of "time"
*       t2id     - netCDF variable id of "time2"
*       t0       - T0 string for the time axis
*       tunit    - number of seconds per axis time unit
*       index    - subscript of time axis to return
*       dval     - double precision value returned
*       status   - TMAP library return status

* argument declarations
      INTEGER cdfid, t1id, t2id, index, status
      REAL tunit
      REAL*8  dval
      CHARACTER*(*) t0

#ifdef unix
      include 'tmap_format/netcdf.decl'
      include 'tmap_format/netcdf.inc'
      include 'tmap_format/tmap_errors.parm'
#else
      INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
      INCLUDE 'TMAP_FORMAT:NETCDF.INC'
      INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
#endif

* internal variable declarations:
      INTEGER jday, msec, cdfstat, mon, day, yr, hour, min, sec,
     .        t0year, t0mon, t0day, t0hour, t0min, t0sec
      REAL*8          TM_SECS_FROM_BC
      REAL*8          date_secs, start_secs, offset_secs

* break apart the T0 offset date
        CALL TM_BREAK_DATE (t0, t0year, t0mon,
     .                      t0day, t0hour, t0min, t0sec, status)

* read the 2 integer values
      CALL NCVGT1( cdfid, t1id, index, jday, cdfstat )
      IF ( cdfstat .NE. NCNOERR ) CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_RD_EP_TIME', cdfid, t1id,
     .       'EPIC time', no_errstring, *5000 )
      CALL NCVGT1( cdfid, t2id, index, msec, cdfstat )
      IF ( cdfstat .NE. NCNOERR ) CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_RD_EP_TIME', cdfid, t2id,
     .       'EPIC time2', no_errstring, *5000 )

* convert the 2 EPIC integers to a date
      CALL TM_EP_TIME_CONVRT( jday, msec,
     .                        mon, day, yr, hour, min, sec)

* determine seconds from BC for each part
      start_secs = TM_SECS_FROM_BC (t0year, t0mon, t0day,
     .                              t0hour, t0min, t0sec)
      date_secs  = TM_SECS_FROM_BC (yr  , mon, day,
     .                              hour, min, sec)

* Find the number of seconds since T0
      offset_secs = date_secs - start_secs

* Divide by time step increment to get # of time step
      dval = offset_secs/tunit

      status = merr_ok
      RETURN

* error exit
 5000 RETURN

      END


