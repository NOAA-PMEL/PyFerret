	SUBROUTINE MAKE_AXIS( grid, idim, lo_ss, hi_ss, axis )

* create a REAL*4 array of world coordinates over a range of subscripts along
* axis IDIM of grid GRID.  Begin the axis at subscript 1 of AXIS.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/20/88
* revision 0.1 -11/28/88 - new TM IO library: WORLD --> TM_WORLD

#ifdef unix
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	INTEGER	grid, idim, lo_ss, hi_ss
	REAL    axis(*)

* internal variable declarations:
	INTEGER	ito, i
	REAL*8	TM_WORLD

	ito = 0

	DO 100 i = lo_ss, hi_ss
	   ito = ito + 1
	   axis(ito) = TM_WORLD( i, grid, idim, box_middle )
 100	CONTINUE

	RETURN
	END
