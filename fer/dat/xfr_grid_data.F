	BLOCK DATA XFR_GRID_DATA

* initialize FERRET extensions to TMAP_FORMAT grid data

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  7/13/89

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'ferret_cmn/xfr_grid.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'FERRET_CMN:XFR_GRID.CMN'
#endif

	DATA	grd_stk_ptr / max_grids /

	END
