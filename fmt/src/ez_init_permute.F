	SUBROUTINE EZ_INIT_PERMUTE(varnum )

* work routine for EZ_READ when /ORDER is not the default X-Y-Z-T
* Initialize the variables used to translate the permuted subscripts
* into the 1D array index

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Thermal Modeling and Analysis Program

* revision 0.00 -  5/94 *sh*
* Linux port -kob - 3/97 - Modify include for tmap_dset.parm to be a
*			   preprocessor include for F90 - needed because
*			   tmap_dset.parm contains an ifdef

* calling arguments:
* varnum - the variable number in COMMON
	INTEGER varnum

#ifdef unix
        include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
        include 'tmap_format/gt_lib.parm'
        include 'tmap_format/xtm_grid.cmn_text'
        include 'tmap_format/xdset_info.cmn_text'
        include 'tmap_format/xez_order.cmn_text'
#else
        INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
        INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
        INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
        INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN_TEXT'
        INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN_TEXT'
	INCLUDE 'TMAP_FORMAT:XEZ_ORDER.CMN'
#endif

* local variable definitions:
	INTEGER idim, gridnum, line

* initialize
	gridnum = ds_grid_number(varnum)

* ezii    - the "current" subscript, zero referenced
* ezorder - the permuted ordering
* eznn    - the length of each axis
	DO 100 idim = 1, 4
	   ezii(idim) = 0
	   ezorder(idim) = ds_ordering(idim, varnum)
	   line = grid_line(idim, gridnum)
	   IF ( line .EQ. mpsnorm) THEN
	      eznn(idim) = 1
	   ELSE
	      eznn(idim) = line_dim(line)
	   ENDIF
 100	CONTINUE

* decrement by 1 since EZ_READ pre-increments
	ezii(ezorder(1)) = ezii(ezorder(1)) - 1
	RETURN
        END
