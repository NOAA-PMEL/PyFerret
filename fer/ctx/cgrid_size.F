	INTEGER FUNCTION CGRID_SIZE ( cx )

* determine the number of values in a 2D grid that lies within the plane
* of the specified context

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/24/86
* revision 0.1 - 7/20/87 - kludge introduction of 1D grids for XY averages
* V200:  5/18/89 - 4D symmetrical

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

	INTEGER	CX_DIM_LEN, cx

	CGRID_SIZE = CX_DIM_LEN( 1, cx )
     .		   * CX_DIM_LEN( 2, cx )
     .		   * CX_DIM_LEN( 3, cx )
     .		   * CX_DIM_LEN( 4, cx )

	RETURN
	END
