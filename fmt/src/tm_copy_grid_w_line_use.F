	SUBROUTINE TM_COPY_GRID_W_LINE_USE ( source, dest )
*
* Copies grid information from source to dest
* and passes along responsibility for associated dynamic lines

* 10/95 for Ferret V4.20
*
* Arguments
	INTEGER*4	source, dest
*
* Parameters and commons
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif
*
* Local declarations
	INTEGER*4	icnt
*

* copy the grid
	CALL TM_COPY_GRID( source, dest )

* give the new grid responsibility for dynimic lines it has inherited
	DO 100 icnt = 1,4
	  IF ( grid_line(icnt,dest) .GT. max_lines ) CALL
     .		TM_USE_LINE( grid_line(icnt,dest) )
  100	CONTINUE
	RETURN
	END
