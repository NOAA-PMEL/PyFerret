	SUBROUTINE TRANSFER_GRID ( source_grid, dest_grid )

* transfer a complete description of a grid from one slot in COMMON to another

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 10/14/86
* revision 0.1 -  3/23/87 - eliminated grid_origin as not useful

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

	INTEGER	source_grid, dest_grid, idim

	grid_name     ( dest_grid )	= 'unknown'
	grid_rotation ( dest_grid )	= grid_rotation( source_grid )

	DO 100 idim = 1, 4
	   grid_line    (idim, dest_grid) = grid_line    (idim, source_grid)
	   grid_out_prod(idim, dest_grid) = grid_out_prod(idim, source_grid)
 100	CONTINUE

	RETURN
	END
