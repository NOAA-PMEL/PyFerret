	SUBROUTINE GRID_WORLD_EXTREMES ( lo, hi, grid, idim )

* find the extreme bounds of the indicated grid on the indicated axis
* in WORLD coordinates
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 10/16/86
* revision 1.0 - 3/10/87  - changed to axis data stored in COMMON/~VARIABLES/
*			    and changed time axis dependence
* revision 1.1 - 2/18/88  - incorporated "regular" axes
* revision 1.2 - 20/12/88 - incorporated new TM IO library (no box sizes)
* revision 1.3 - 2/6/89   - added modulo axes of arbitrary length
* Unix/RISC port - 7/91   - eliminated unneeded *variables common
* V4.20 - 10/95 - remove references to line_mem to permit recursive line defs

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
	REAL*8	lo,hi

* local variable declarations
	INTEGER line_num, line_len
	REAL*8	TM_WORLD

	line_num = grid_line( idim, grid )
	line_len = line_dim( line_num )

	IF     ( line_num .EQ. mnormal  ) THEN

	   lo = unspecified_val8
	   hi = unspecified_val8

	ELSEIF ( line_num .EQ. munknown ) THEN

	   lo = arbitrary_small_val8
	   hi = arbitrary_large_val8

	ELSEIF ( line_modulo(line_num) ) THEN

	   lo = arbitrary_small_val8
	   hi = arbitrary_large_val8

	ELSE

	   IF ( line_regular( line_num ) ) THEN
	      lo = line_start( line_num ) - line_delta( line_num )/2.
	      hi = lo + line_delta( line_num ) * line_dim( line_num )

	   ELSE
* ... lo limit of bottom box; hi limit of top box
	      lo = TM_WORLD(       1, grid, idim, box_lo_lim)
	      hi = TM_WORLD(line_len, grid, idim, box_hi_lim)


	   ENDIF

	ENDIF

	RETURN
	END

