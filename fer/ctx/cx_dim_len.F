	INTEGER FUNCTION CX_DIM_LEN ( idim, cx )

* determine the number of points along the indicated axis of a variable context

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/19/87
* revision 0.1 - 7/8/87  - checks transformations
* V200:  5/18/89 - 4D-symmetrical - transforms reflected in subscript limits

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	idim, cx

	   CX_DIM_LEN = cx_hi_ss( cx, idim ) 
     .		      - cx_lo_ss( cx, idim ) + 1

	RETURN
	END
