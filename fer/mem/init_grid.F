	SUBROUTINE INIT_GRID ( grid, name, init_ax )

* initialize the definition of a grid with non-commital information

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V420 9/8/95 - for implicit grid creation

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations
	CHARACTER*(*) name
	INTEGER	grid, init_ax

* internal variable declarations
	INTEGER	idim

	grid_name     ( grid )	= name
	grid_rotation ( grid )	= 0.0

	DO 100 idim = 1, 4
	   grid_line    (idim, grid) = init_ax
	   grid_out_prod(idim, grid) = .TRUE.
 100	CONTINUE

	RETURN
	END
