	SUBROUTINE TM_COPY_GRID ( source, dest )
*
* Copies grid information from source to dest
*
* revision 0.00 - 11/22/88
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
	grid_name(dest) = grid_name(source)
	DO 100 icnt = 1,4
	  grid_line(icnt,dest) = grid_line(icnt,source)
	  grid_out_prod(icnt,dest) = grid_out_prod(icnt,source)
  100	CONTINUE
	grid_rotation(dest) = grid_rotation(source)
	RETURN
	END
