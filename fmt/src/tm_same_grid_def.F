	LOGICAL FUNCTION TM_SAME_GRID_DEF( grid1, grid2 )

* Check to see if the guts of the two grids are identical

* revision 0.00 - Steve Hankin - based on TM_FIND_LIKE_GRID by  M. Verschell
* 12/3/92 *sh* - on mpsnorm axes ignore inner/outer
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

	DO 100 i=1,4
	  IF (grid_line(i,grid1) .NE. grid_line(i,grid2) ) GOTO 1000
          IF ( grid_line(i,grid1) .NE. mpsnorm ) THEN           ! 12/3
             IF (grid_out_prod(i,grid1) .NEQV. grid_out_prod(i,grid2))
     .	                                                   GOTO 1000
          ENDIF
  100	CONTINUE

* All 4 axis matched - check grid rotation
	IF (grid_rotation(grid1) .NE. grid_rotation(grid2)) GOTO 1000

* they match
  	TM_SAME_GRID_DEF = .TRUE.
        RETURN

* they don't match
 1000   TM_SAME_GRID_DEF = .FALSE.
        RETURN

	END
