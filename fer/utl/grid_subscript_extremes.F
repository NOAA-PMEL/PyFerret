	SUBROUTINE GRID_SUBSCRIPT_EXTREMES ( lo, hi, grid, idim )

* find the extreme bounds of the indicated grid on the indicated axis
* in SUBSCRIPT values

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 10/16/86
* revision 0.1 - 3/23/87 - added new TMAP library: time axes, file var. names
* revision 0.2 - 2/6/89  - added modulo axes of arbitrary length

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations
	INTEGER	grid, idim
	INTEGER	lo,hi

* local variable declarations
	INTEGER line_num

	line_num = grid_line( idim, grid )

	IF     ( line_num .EQ. mnormal  ) THEN

	   lo = unspecified_int4
	   hi = unspecified_int4

	ELSEIF ( line_num .EQ. munknown ) THEN

	   lo = unspecified_int4
	   hi = unspecified_int4

	ELSEIF ( line_modulo(line_num) ) THEN

	   lo = arbitrary_small_int4
	   hi = arbitrary_large_int4

	ELSE

	   lo = 1
	   hi = line_dim(line_num)

	ENDIF

	RETURN
	END

