	CHARACTER*(*) FUNCTION CD_CHILDAX_NAME( axname, lo, hi, tlen )

* Determine the name to use for a "child axis" (sub-region) of a parent axis

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92
* 1/95 - allow negative subscripts.  Substitute "N" for "-"

* argument definitions:
*     axis    - original axis name
*     lo      - lo subscript
*     hi      - hi subscript
*     slen    - returned length of name

#ifdef unix
        include 'tmap_format/netcdf.decl'
        include 'tmap_format/netcdf.inc'
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
xx        include 'xxtmap_format/netcdf.decl'  ! untested
xx        include 'xxtmap_format/netcdf.inc'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* argument declarations
      INTEGER lo, hi, tlen
      CHARACTER*(*) axname

* local variable declarations:
      INTEGER TM_LENSTR1, slen
      CHARACTER TM_LEFINT*8, int_buff*8

* generate "axis_nameLO_HI"
* append the low subscript limit
      tlen = TM_LENSTR1(axname)
      int_buff = TM_LEFINT(lo,slen)
      IF (int_buff(1:1) .EQ. '-') int_buff(1:1) = 'N'
      CD_CHILDAX_NAME = axname(:tlen)//int_buff
      tlen = tlen+slen

* append the high subscript limit
      int_buff = TM_LEFINT(hi,slen)
      IF (int_buff(1:1) .EQ. '-') int_buff(1:1) = 'N'
      CD_CHILDAX_NAME = CD_CHILDAX_NAME(:tlen)//'_'//int_buff
      tlen = tlen+slen+1

      RETURN
      END
