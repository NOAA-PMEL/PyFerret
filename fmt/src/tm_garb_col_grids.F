	SUBROUTINE TM_GARB_COL_GRIDS ( first, last )

* garbage collect wasted space in newly defined grids

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92

* argument definitions:
*       first   - highest number grid to check
*       last    - lowest number grid to check
* note: grids are allocated from high to low

* include files
#ifdef unix
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
        include 'tmap_format/xdset_info.cmn_text'
        external xdset_info_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
        INCLUDE 'TMAP_FORMAT/XDSET_INFO.CMN'
#endif

* calling argument declarations:
      INTEGER  first, last

* local variable declarations:
      INTEGER igrid, higrid, i

* initialize grid loop
!  like: DO 200 igrid = first, last  but last keeps shifting down
      igrid = first
      higrid = last

* is this a hole ?
 10   IF ( grid_name(igrid) .NE. char_init ) GOTO 300

* modify grid pointers saved in the variables
           DO 100 i = 1, maxvars
              IF ( ds_grid_number(i) .LT. igrid
     .       .AND. ds_grid_number(i) .GE. higrid ) 
     .                ds_grid_number(i) = ds_grid_number(i) - 1
 100       CONTINUE

* shift all the definitions back to fill the hole
           DO 200 i = igrid+1, higrid
 200       CALL TM_COPY_GRID( i, i-1 )

* initialize new hole at end
           grid_name(higrid) = char_init
           higrid = higrid - 1
           igrid = igrid - 1    ! to nullify decrement, below

! like: 300    CONTINUE
 300    igrid = igrid + 1
        IF ( igrid .LE. higrid ) GOTO 10

* successful completion
        RETURN

	END
