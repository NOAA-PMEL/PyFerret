	LOGICAL FUNCTION ITSA_MODULO_AXIS( cx, idim )

* determine whether a particular context grid axis is a modulo one

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
	INTEGER	cx, idim

* internal variable declarations:
	INTEGER axis

	axis = grid_line(idim, cx_grid(cx))

	ITSA_MODULO_AXIS = line_modulo(axis)

	RETURN
	END
