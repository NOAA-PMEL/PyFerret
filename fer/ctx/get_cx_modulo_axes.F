	SUBROUTINE GET_CX_MODULO_AXES( cx, modulo )

* get the modulo status of all 4 axes of a context grid
* ... all 4 at once for performance reasons -- so this is called 1/4 as often

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420: 11/95

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/xtm_grid.cmn_text'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN_TEXT'
#endif

* calling argument declarations:
	INTEGER	cx
	LOGICAL modulo(4)

* internal variable declarations:
	INTEGER	grid, idim

	grid = cx_grid(cx)

	DO 100 idim = 1, 4
 100 	modulo(idim) = line_modulo(grid_line(idim,grid))	

	RETURN
	END
