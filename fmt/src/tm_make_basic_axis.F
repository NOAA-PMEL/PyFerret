	SUBROUTINE TM_MAKE_BASIC_AXIS( name, start,
     .                                 delta, npts, iaxis, status )

* Create a very basic 1 to n axis with no frills

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92

* argument definitions:
*       name      - name for axis
*       start     - starting value
*	delta     - delta value
*       npts      - number of points
*       iaxis     - returned axis pointer in TMAP common
*	status	  - return status

* include files
#ifdef unix
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* argument declarations
	INTEGER	 npts, iaxis, status
        REAL*8   start, delta
        CHARACTER*(*) name

* local variable declarations:
      INTEGER TM_FIND_LINE_SLOT, STR_UPCASE,
     .        istat

* locate storage space for the axis definition
      status = TM_FIND_LINE_SLOT ( iaxis )
      IF (status .NE. merr_ok) GOTO 5900

* build axis
      istat = STR_UPCASE( line_name(iaxis), name )
      line_dim( iaxis)      = npts
      line_start(iaxis)     = start
      line_delta(iaxis)     = delta
      line_regular(iaxis)   = .TRUE.
      line_units(iaxis)     = ' '
      line_unit_code(iaxis) = 0
      line_modulo(iaxis)    = .FALSE.
      line_t0(iaxis)        = char_init20
      line_tunit(iaxis)     = real4_init
      line_direction(iaxis) = 'NA'

* successful completion
      status = merr_ok

 5900 RETURN

	END
