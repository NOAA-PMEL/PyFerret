	SUBROUTINE TM_GET_LIKE_GRID ( newgrid, newname, gridnum, status)
*
* Will compare "newgrid" to other grid definitions in common. If there is a
* match then "gridnum" returns with the number of the match. If there is no
* match, then a new slot is located and the name "newname" is used to name it,
* and "gridnum" returns with the number of the new slot.
*
* revision 0.00 - 11/22/88
*
* Arguments
	INTEGER*4	newgrid, gridnum, status
	CHARACTER*(*)	newname
*
* INCLUDE FILES
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif
*
* Local definitions
	INTEGER*4	grid_match
	INTEGER*4	TM_FIND_LIKE_GRID, TM_FIND_GRID_SLOT
*
* Check for match
	grid_match = TM_FIND_LIKE_GRID ( newgrid )
	IF (grid_match .NE. unspecified_int4) THEN
	  gridnum = grid_match
	  GOTO 9000
	ENDIF
*
* No match, find empty slot for storage
	status = TM_FIND_GRID_SLOT ( gridnum )
	IF (status .NE. merr_ok) GOTO 9000
*
* Move everything over
	CALL TM_COPY_GRID( newgrid, gridnum )
*
* Name the new grid
	grid_name(gridnum) = newname
*
	status = merr_ok
 9000	RETURN
	END
