	SUBROUTINE ADD_C_2_GRID ( length, grid_data, addend )

* add a constant to a grid

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 6/25/87

#ifdef unix
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	INTEGER	length
	REAL	grid_data( length ), addend

* internal variable declarations:
	INTEGER	i
		
	DO 100 i = 1, length
 100	IF (  grid_data(i) .NE. bad_val4 )
     .			grid_data(i) = grid_data(i) + addend

	RETURN
	END

