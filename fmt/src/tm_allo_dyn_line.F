	SUBROUTINE TM_allo_dyn_line( line, status )

* Allocate a line from the dynamic line area (above max_lines)

* The lines in xtm_grid.cmn fall into two categories (ideally they could
* and should all be category 2, but the development was historical)
* category 1: explicit (static) lines
*	these are located in the low indices of the grid common
*	(e.g. line_name(3)).  They are created by READ_GRIDFILE, CD_*_GRIDS,
*	and by DEFINE AXIS inside Ferret.  These are the lines that are
*	returned by TM_FIND_LINE_SLOT.
* category 2: dynamic lines
*	these are in the upper indices: above max_lines.  They are managed
*	efficiently with linked lists and usage counts.  They are used by
*	Ferret to hold "implicit" lines -- implied by expressions such as
*	SALT[GX=lo:hi:del] where an X axis is implied.
*	They are deleted as necessary to ensure that new line definitions and
*	changes in user variable definitions do not corrupt memory. 
* Note: there is a difference in Ferret between the management of dynamic
* grids and axes.  Namely, **ALL GRIDS** are regarded as a single "grid pool"
* including both dynamic and static: when a static grid exists that matches
* a need it will be used in preference to making a dynamic grid.  This is done
* to ensure consistency -- at all stages of variable evaluation a given
* must be associated with the same grid.  With lines, however, the challenge
* is tracking "who" is using any given line.  Since this is not conveniently
* knowable (too compute intensive to figure out every time it is of interest)
* a greater separation is created between the static and dynamic lines.  This
* is reflected in the differences between TM_GET_LIKE_DYN_LINE and
* TM_GET_LIKE_DYN_GRID.

* Steve Hankin  NOAA/PMEL/TMAP

* 10/3/95 for Ferret version 4.2

* Include files
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	include 'tmap_format/xio.cmn_text'
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'TMAP_FORMAT:XIO.CMN_TEXT'
#endif

* Argument definitions
	INTEGER line, status

* Local definitions
	INTEGER next_free_line

* take a line from the "free" list and put it on the "used" list
	IF ( line_free_ptr .NE. 0 ) THEN
* ... unhook from free list
	   line = line_free_ptr
	   next_free_line  	= line_flink(line)
	   line_free_ptr	= next_free_line
* ... hook onto front of used list
	   line_flink(line)	= line_flink(max_lines)
	   line_blink(line)	= max_lines
	   line_flink(max_lines)= line
	   line_blink(line_flink(line))	= line
	ELSE
	   CALL TM_ERRMSG (merr_linelim, status,
     .                  'TM_ALLO_DYN_LINE', no_descfile, no_stepfile,
     .                  '>num_dyn_lines dynamic lines',
     .                  no_errstring, *9999)
	ENDIF

* flag it as in NOT use - its use count gets bumbed when used by a grid or
* a child axis
	line_use_cnt(line) = 0

* successful completion
	status = merr_ok
 9999	RETURN

	END
