	LOGICAL FUNCTION NO_GRID_RANGE( grid, n, mg, val1 )

* determine if all the data indicated are of the same value

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/2/88
* V200:  5/19/89 - 4D symmetrical

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	n, mg
	REAL 	val1, grid( n )

* internal variable declarations:
	INTEGER	i
	REAL	bad_flag

* initialize
	bad_flag = mr_bad_data( mg )

* comparison value (ends up as first non-bad value)
	val1 = bad_flag

* check all data within the indicated context
	DO 100 i = 1, n
	   IF ( grid(i) .NE. val1 ) THEN
	      IF     ( val1       .EQ. bad_flag ) THEN
	         val1 = grid(i)
	      ELSEIF ( grid(i) .NE. bad_flag ) THEN
	         NO_GRID_RANGE = .FALSE.
	         RETURN
	      ENDIF
	   ENDIF
 100	CONTINUE

* they were all identical - val1=bad_flag if they were all bad
	NO_GRID_RANGE = .TRUE.
	RETURN
	END
