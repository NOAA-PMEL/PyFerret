	CHARACTER*(*) FUNCTION CD_AXIS_NAME ( grid, idim, nlen )

* determine the netCDF name to use for the indicated axis of this grid

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 6/92
* V4.20  11/95 - remove parentheses surrounding dynamic axis names

* argument definitions:
*     grid    - TMAP grid id
*     idim    - axis orientation (1-4)
*     nlen    - length of name

* include files
#ifdef unix
	include 'tmap_format/tmap_errors.parm'
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

* argument declarations
      INTEGER grid, idim, nlen

* local variable declarations:
      LOGICAL TM_ABSTRACT_AXIS
      INTEGER TM_LENSTR1, iaxis

      CHARACTER*1 ww_dim_name(4)
      DATA ww_dim_name/'X', 'Y', 'Z', 'T'/

* initialize
      iaxis = grid_line(idim, grid)

* basic name straight from the grid
      CD_AXIS_NAME = line_name(iaxis)
      nlen = TM_LENSTR1( CD_AXIS_NAME )

* if it's an abstract axis make a special name
      IF ( TM_ABSTRACT_AXIS( iaxis ) ) THEN
         CD_AXIS_NAME = ww_dim_name(idim)//'AX'
         nlen = 3

* if it's a dynamic axis remove the parentheses
      ELSEIF ( line_name(iaxis)(1:1) .EQ. '(' ) THEN
	 IF ( nlen .LE. 2) nlen = 3			! should never happen
         CD_AXIS_NAME = line_name(iaxis)(2:nlen-1)
	 nlen = nlen - 2
      ENDIF

      RETURN
      END
