	SUBROUTINE CD_ABORT_GRIDS

* Wipe out partially created grids and axes - a fatal error has occurred

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
* Linux port -kob - 3/97 - Modify include for tmap_dset.parm to be a
*			   preprocessor include for F90 - needed because
*			   tmap_dset.parm contains an ifdef

* include files
#ifdef unix
        include 'tmap_format/xbuild_grids.cmn'
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xtm_grid.cmn_text'
#else
        INCLUDE 'TMAP_FORMAT:XBUILD_GRIDS.CMN'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* local variable declarations:
      INTEGER igrid, iaxis

* restore grid names to their "unused" markers
 	DO 200 igrid = firstgrid, lastgrid
 200    grid_name(igrid) = char_init

* restore axis names to their "unused" markers
        DO 300 iaxis = firstax, lastax
	    line_name  (iaxis) = char_init
	    line_subsc1(iaxis) = unspecified_int4
 300    CONTINUE

* return line memory to unused state
	next_line_mem_pos = orig_line_mem

        RETURN
	END
