	SUBROUTINE EQUAL_RANGE_T( string,idim,grid,lo,hi,delta,status )

* from a string of the form "name=value", return the lo/high value range
* if the string is simply "name" then return unspecified values
* the "delta" value is optional - return unspecfied_val8 if not given

* This routine differs from EQUAL_RANGE in the treatment of time axes.
* This routine always returns a tstep value.  If MODE CALENDAR is TRUE
* this routine checks to make sure the points are ordered.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V3.00  20/10/93 - *sh*

* calling argument declarations
	INTEGER		idim, grid, status
	REAL*8		lo, hi, delta
	CHARACTER*(*)	string

* internal variable declarations
	LOGICAL		GEOG_LABEL, formatted
	REAL*8		SECS_TO_TSTEP, tmp

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
#endif

* first get EQUALL_RANGE to convert the data normally
	CALL EQUAL_RANGE( string, idim, lo, hi, delta,
     .			  formatted, status )
	IF ( status .NE. ferr_OK ) RETURN

	IF ( idim .EQ. t_dim ) THEN
* if the time was given as a date convert it to a tstep
	   IF ( formatted ) THEN
	      lo = SECS_TO_TSTEP( grid, lo )
	      hi = SECS_TO_TSTEP( grid, hi )
	   ENDIF		

* if MODE CALENDAR is TRUE in FERRET make sure the times are ordered
	   IF ( GEOG_LABEL(t_dim,grid)
     .    .AND. lo .GT. hi		 ) THEN 
	      tmp = lo
	      lo = hi
	      hi = tmp
	   ENDIF
	ENDIF

* success
 1000	RETURN
	END
