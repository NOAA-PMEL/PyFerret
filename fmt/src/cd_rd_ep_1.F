	SUBROUTINE CD_RD_EP_1(cdfid,t1id,t2id,t0,tunit,index,dval,status)

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


