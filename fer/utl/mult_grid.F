	SUBROUTINE MULT_GRID ( length, grid_data, factor )

* multiply a grid by a constant factor

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/23/86

#ifdef unix
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

	INTEGER	length, i
	REAL	grid_data( length ), factor
	
	DO 100 i = 1, length
 100	IF (  grid_data(i) .NE. bad_val4 )
     .			grid_data(i) = grid_data(i) * factor

	RETURN
	END

