	INTEGER FUNCTION TM_DEALLO_DYN_LINE_SUB( line )

* Deallocate a line from the dynamic line area (above max_lines)
* Called quasi-recursively by tm_deallo_dyn_line

* Steve Hankin  NOAA/PMEL/TMAP

* 10/3/95 for Ferret version 4.2

* Include files
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'tmap_format/xio.cmn_text'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'TMAP_FORMAT:XIO.CMN_TEXT'
#endif

* Argument definitions
	INTEGER line

* Local definitions
	INTEGER next_used_line

* initialize
	TM_DEALLO_DYN_LINE_SUB = 0

* must be a dynamic line
	IF ( line .LE. max_lines ) THEN
	   CALL TM_NOTE('Intern err: TM_DEALLO_DYN_LINE:1 !!!', lunit_errors)
	   RETURN
	ENDIF

* decrement the used count for this line
	line_use_cnt(line) = line_use_cnt(line) - 1

* does anyone still have a claim on this line?
	IF ( line_use_cnt(line) .GT. 0 ) THEN
	   RETURN

* a use count of -1 results from a line which was allocated and then promptly
* deallocated without being used
	ELSEIF ( line_use_cnt(line) .GE. -1 ) THEN
* take this line from the "used" list and put it on the "free" list
* ... hook onto free list
	   next_used_line   = line_flink(line)
	   line_flink(line) = line_free_ptr  
	   line_free_ptr    = line
	   
* ... unhook from used list
	   line_flink(line_blink(line)) = next_used_line
	   line_blink(next_used_line)   = line_blink(line)

	ELSE
* internal error -- the use count is negative!!
	   CALL TM_NOTE('Intern err: TM_DEALLO_DYN_LINE:2 !!!', lunit_errors)
	ENDIF

* return flag to indicate if the quasi-recursive descent ends here
* ... zero or index of parent axis to deallocate
	TM_DEALLO_DYN_LINE_SUB = line_parent(line)

* successful completion
	RETURN

	END
