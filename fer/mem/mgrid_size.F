	INTEGER FUNCTION MGRID_SIZE ( mr )

* determine the number of values in a 2D grid that lies within the plane
* of the specified variable stored in memory

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/24/86
* revision 0.1 - 7/20/87 - kludge introduction of 1D grids for XY averages
* V200:  5/18/89 - 4D symmetrical

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

	INTEGER	MR_DIM_LEN, mr

	   MGRID_SIZE = MR_DIM_LEN( 1, mr )
     .		      * MR_DIM_LEN( 2, mr )
     .		      * MR_DIM_LEN( 3, mr )
     .		      * MR_DIM_LEN( 4, mr )

	RETURN
	END
