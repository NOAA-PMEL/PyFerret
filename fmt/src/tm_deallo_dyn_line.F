	SUBROUTINE TM_DEALLO_DYN_LINE( line )

* Deallocate a line from the dynamic line area (above max_lines)

* Steve Hankin  NOAA/PMEL/TMAP

* 10/3/95 for Ferret version 4.2

* Since FORTRAN is not recursive this routine implements a poor man's
* recursion so that the deallocation can trace down the lineage of
* parent lines deallocating the parents if the child gets snuffed out.

* Argument definitions
	INTEGER line

* Local definitions
	INTEGER TM_DEALLO_DYN_LINE_SUB, gonner

* Include files
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/xtm_grid.cmn_text'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* initialize
	gonner = line

* quasi-recursive loop
 10	gonner = TM_DEALLO_DYN_LINE_SUB( gonner )
	IF (gonner.GT.max_lines .AND. gonner.NE.0) GOTO 10

* successful completion
	RETURN
	END
