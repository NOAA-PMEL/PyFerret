	INTEGER FUNCTION MR_DIM_LEN ( idim, mr )

* determine the no. of points along the indicated axis of a memory resident var.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  5/18/89 - 4D-symmetrical - transforms reflected in subscript limits

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER	idim, mr

	   MR_DIM_LEN = mr_hi_ss( mr, idim ) 
     .		      - mr_lo_ss( mr, idim ) + 1

	RETURN
	END
