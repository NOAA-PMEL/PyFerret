	SUBROUTINE CD_RD_YMD_ARR(cdfid,t1id,t0,tunit,n,darr,status)

* read an array of time values from a "yymmddhhmmss" time axis
* if T0 is not yet known determine an appropriate value
* return time values as a double precision time relative to t0

* NOTE:  This routine could be grestly optimized for performance.
*     This version calls CD_RD_ARR_1 for each point.
*     Instead it could break apart T0 only once and use array reads to read
*     The double integers directly into the returned double precision array
*     The integers could then be re-arranged into pairs and converted to 
*     TMAP times in place.

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/95

* argument definitions:
*	cdfid	 - netCDF id number for already opened CDF file
*       t1id     - netCDF variable id of "time"
*       t0       - T0 string for the time axis
*       tunit    - number of seconds per axis time unit
*       n        - number of time values to return
*       darr     - double precision array returned
*       status   - TMAP library return status

* argument declarations
      INTEGER cdfid, t1id, n, status
      REAL tunit
      REAL*8  darr(n)
      CHARACTER*(*) t0

#ifdef unix
      include 'tmap_format/tmap_errors.parm'
#else
      INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
#endif

* internal variable declarations:
      INTEGER i

      DO 100 i = 1, n
 100  CALL CD_RD_YMD_1(cdfid,t1id,t0,tunit,i,darr(i),status)

      RETURN

      END


