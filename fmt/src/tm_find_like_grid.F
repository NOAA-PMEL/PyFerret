	INTEGER*4 FUNCTION TM_Find_like_grid ( gridnum )
*
* Check to see if an existing grid is identical to the grid pointed to by
* "gridnum".
*
* by M. Verschell PMEL/TMAP
*
* revision 0.00 - 05/18/88
* revision 0.01 - 11/21/88 - variable used check changed to specific char_initXX
* revision 1.0 - 2/92 *sh* - removed guts to TM_SAME_GRID_DEF
*
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
	INTEGER*4   gridnum

* Local definitions
        LOGICAL     TM_SAME_GRID_DEF
	INTEGER*4   j

* Loop through the grids comparing the axis names
	DO 100 j = 1, max_grids
	  IF (grid_name(j).NE.char_init16 .AND. j.NE.gridnum) THEN
             IF ( TM_SAME_GRID_DEF( gridnum, j ) ) THEN
 	        TM_Find_like_grid = j
                RETURN
             ENDIF
          ENDIF
  100	CONTINUE

* matching grid doesn't exist
	TM_Find_like_grid = unspecified_int4
	RETURN

	END
