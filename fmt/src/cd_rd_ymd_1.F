	SUBROUTINE CD_RD_YMD_1(cdfid,t1id,t0,tunit,index,dval,status)

* read a single time value from a "yymmddhhmmss" time axis
* if T0 is not yet known determine an appropriate value
* return time value as a double precision time relative to t0

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/95  -- specifically for accessing CDC files

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
      INTEGER cdfstat, mon, day, yr, hour, min, sec,
     .        t0year, t0mon, t0day, t0hour, t0min, t0sec
      REAL*8  TM_SECS_FROM_BC
      REAL*8  yymmdd, date_secs, start_secs, offset_secs

* internal parameter declarations
      REAL*8      p1900, p0010
      PARAMETER ( p1900 = 693961. * 24 * 60 * 60,
     .		  p0010 = 3653.   * 24 * 60 * 60 )

* read the yymmdd value
      CALL NCVGT1( cdfid, t1id, index, yymmdd, cdfstat )
      IF ( cdfstat .NE. NCNOERR ) CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_RD_EP_TIME', cdfid, t1id,
     .       'yymmdd time', no_errstring, *5000 )

* convert the 2 EPIC integers to a date
      CALL TM_YMD_TIME_CONVRT( yymmdd,
     .                         mon, day, yr, hour, min, sec)

* determine seconds from BC
      date_secs  = TM_SECS_FROM_BC (yr  , mon, day,
     .                              hour, min, sec)

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
      CALL TM_BREAK_DATE (t0, t0year, t0mon,
     .                    t0day, t0hour, t0min, t0sec, status)

      start_secs = TM_SECS_FROM_BC (t0year, t0mon, t0day,
     .                              t0hour, t0min, t0sec)

* Find the number of seconds since T0
      offset_secs = date_secs - start_secs

* Divide by time step increment to get # of time step
      dval = offset_secs/tunit

      status = merr_ok
      RETURN

* error exit
 5000 RETURN

      END


