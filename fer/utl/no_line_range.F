	LOGICAL FUNCTION NO_LINE_RANGE( array, npts, bad_flag, val1 )

* determine if all the data indicated are of the same value
* if true val1=bad_flag if all points are bad-data flags

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/2/88
* revision 0.1 - 9/22/88 - bug fix.  start at subscript 1

#ifdef unix
	include	'ferret_cmn/ferret.parm'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	INTEGER	npts
	REAL 	array( npts ), bad_flag, val1

* internal variable declarations:
	INTEGER	i

* comparison value
	val1 = bad_flag

* check all data within the indicated context
	DO 100 i = 1, npts
	   IF ( array( i ) .NE. val1 ) THEN
	      IF     ( val1       .EQ. bad_flag ) THEN
	         val1 = array( i )
	      ELSEIF ( array( i ) .NE. bad_flag ) THEN
	         NO_LINE_RANGE = .FALSE.
	         RETURN
	      ENDIF
	   ENDIF
 100	CONTINUE

* they were all identical --> val1 contains bad_flag if they were all bad
	NO_LINE_RANGE = .TRUE.

	RETURN
	END
