	INTEGER FUNCTION GT_FULL_GRID ( grid_num, idim )

* find the full range of the indicated grid on the indicated axis in SUBSCRIPT
* values

* programmer - mark verschell (from GRID_SUBSCRIPT_EXTREMES by steve hankin)
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 04/23/87
* revision 0.10 - 07/19/88 - Simplified grid/line location in common
* revision 0.20 - 11/29/88 - Unknown/normal line check done on grid_line now
*			     and removed references to munknown, mnormal
*
* include files
#ifdef unix
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/tmap_dims.parm'
        include 'tmap_format/xtm_grid.cmn_text'
        external xgt_grid_data
#else
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* calling argument declarations
	INTEGER		grid_num, idim

* local variable declarations
	INTEGER		line_len, line_num, nline

* locate line in common storage
	line_num = grid_line( idim, grid_num )
	line_len = line_dim( line_num )

	IF ( line_num .EQ. mpsunkn .OR. line_num .EQ. mpsnorm) THEN
	  GT_FULL_GRID = 1
	ELSE
	  GT_FULL_GRID = line_len
	ENDIF

	RETURN
	END
