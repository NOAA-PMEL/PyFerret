	INTEGER*4 FUNCTION TM_find_like_dyn_line( target )

* Locate a line from the dynamic line area (above max_lines)
* that matches the target line

* Steve Hankin  NOAA/PMEL/TMAP

* 9/13/95 for Ferret version 4.2

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
	INTEGER target

* Local definitions
	LOGICAL     TM_SAME_LINE_DEF
	INTEGER specimen

* initialize linked list search
	specimen =  line_flink(max_lines)

* compare with the lines in the "used" list
 100	CONTINUE

	IF ( specimen .EQ. max_lines ) THEN
	   TM_Find_like_dyn_line = unspecified_int4	! no match found
	   RETURN

	ELSEIF ( specimen .EQ. target ) THEN	
	   specimen = line_flink(specimen)		! skip over self

	ELSEIF ( TM_SAME_LINE_DEF( target, specimen ) ) THEN
	   TM_Find_like_dyn_line = specimen
	   RETURN					! success

	ELSE
	   specimen = line_flink(specimen)		! keep looking

	ENDIF

	GOTO 100
	END
