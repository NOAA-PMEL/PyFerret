	DOUBLE PRECISION FUNCTION TDEST_WORLD
     .			( subsc, grid, idim, where_in_box )

* convert a subscript value on a grid into a "world" coordinate
* if the axis is a valid time axis then a second level conversion will convert
* the time step value on the source grid into the equivalent time step on a
* destination grid.  That grid must have been previously specified with
* a call to TDST_WORLD_INIT.

* This routine exists to optimize performance in converting between time axes

* The conversion process is simply based on the fact that each time axis is
* a slope (units) and an intercept (T0)
* Let
*	A - absolute time (seconds since BC)
*	U - units (slope)
*	T - axis-relative time
* so that
*	AT0src - is the absolute time of T0 on the source axis
*	AT0dst - equivalently on the destination axis
* Then
*	Tsrc = (A-AT0src) / Usrc
*	Tdst = (A-AT0dst) / Udst
* from which
*	Tdst = Tsrc*(Usrc/Udst) + (AT0src-AT0dst)/Udst 
* or in terms of the COMMON variables
*	Tdst = Tscr*tcnvrt_slope + tcnvrt_intercept

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V312:  6/2/94

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xregrid.cmn'
#else
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:REGRID.CMN'
#endif

* calling argument declarations:
	INTEGER	subsc, grid, idim, where_in_box

* internal variable declarations:
        INTEGER status
	REAL*8	TM_WORLD, ww

* standard conversion into world coordinates
	ww = TM_WORLD( subsc, grid, idim, where_in_box )
	TDEST_WORLD = ww

* special treatment of calendar time axis
	IF ( idim .EQ. t_dim ) THEN
           IF ( tcnvrt_ok  ) THEN
	      IF ( grid .NE. cnvrt_src_grid ) THEN
	         CALL ERRMSG( ferr_internal, status,
     .				'TDEST_WORLD not initialized', *5000 )
	      ELSE	
	         TDEST_WORLD = ww * tcnvrt_slope + tcnvrt_intercept 	         
	      ENDIF
	   ENDIF
	ENDIF		

 5000	RETURN
	END
