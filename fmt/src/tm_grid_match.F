	LOGICAL FUNCTION TM_Grid_match ( grid1, grid2 )
*
* Check to see if grid1 is identical to grid2
*
* by M. Verschell PMEL/TMAP
*
* revision 0.00 - 10/19/88
*
* IBM port 14apr94 *kob* : Logical expressions need to be compared
*			   using NEQV and EQV
* Include files
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

* Argument definitions
	INTEGER*4	grid1, grid2

* Local definitions
	INTEGER*4	i

* Compare the axis names
	DO 100 i=1,4
	  IF (grid_line(i,grid1) .NE. grid_line(i,grid2) .OR.
     .	      (grid_out_prod(i,grid1) .NEQV. grid_out_prod(i,grid2)))
     .	    GOTO 110
  100	CONTINUE

* All 4 axis matched - check grid rotation
	IF (grid_rotation(grid1) .NE. grid_rotation(grid2)) GOTO 110

* A match - axis and rotation
	TM_GRID_MATCH = .TRUE.
	GOTO 9999

* Doesn't match
  110	TM_GRID_MATCH = .FALSE.

 9999	RETURN
	END
