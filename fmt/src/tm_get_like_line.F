	SUBROUTINE TM_GET_LIKE_LINE ( newline, newname, linenum, status)
*
* Will compare "newline" to other line definitions in common. If there is a
* match then "linenum" returns with the number of the match. If there is no
* match, then a new slot is located and the name "newname" is used to name it,
* and "linenum" returns with the number of the new slot.
*
* revision 0.00 - 11/22/88
*
* Arguments
	INTEGER*4	newline, linenum, status
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
	INTEGER*4	line_match
	INTEGER*4	TM_FIND_LIKE_LINE, TM_FIND_LINE_SLOT
*
* Check for match
	line_match = TM_FIND_LIKE_LINE ( newline )
	IF (line_match .NE. unspecified_int4) THEN
	  linenum = line_match
	  GOTO 9000
	ENDIF
*
* No match, find empty slot for storage
	status = TM_FIND_LINE_SLOT ( linenum )
	IF (status .NE. merr_ok) GOTO 9000
*
* Move everything over
	CALL TM_COPY_LINE ( newline, linenum )
*
* Name the new line
	line_name(linenum) = newname
*
	status = merr_ok
 9000	RETURN
	END
