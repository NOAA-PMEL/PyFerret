	INTEGER FUNCTION CGRID_AXIS ( idim, cx )

* return the axis number for the indicated orientation of the grid in the
* given context

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 6/3/86
* Unix/RISC port - 1/91 - cant use "/star" in documentation
* V420 11/95 - documentation change and clean-up

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

	INTEGER	idim, cx

* find the line number for the indicated axis of this grid
	CGRID_AXIS = grid_line( idim, cx_grid(cx) )

	RETURN
	END


