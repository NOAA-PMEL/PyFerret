	INTEGER FUNCTION CAXIS_LEN ( idim, cx )

* return the length of the axis from a particular context

* this routine is to minimize the need for TMAP COMMON areas in FERRET code

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V320 - 12/94

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

* calling argument declarations
	INTEGER	idim, cx

* internal variable declarations:
	INTEGER line

* find the line number for the indicated axis of this grid
	line = grid_line ( idim, cx_grid( cx ) )

	IF ( line .EQ. mnormal ) THEN
	   CAXIS_LEN = 1
	ELSE
	   CAXIS_LEN = line_dim( line )
	ENDIF

	RETURN
	END
