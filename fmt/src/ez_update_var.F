	SUBROUTINE EZ_UPDATE_VAR( var )
*
* Subroutine will update the XDSET and XSTEP common entries for a single
* variable of an EZ_ data set
*
* revision 0.00 - 10/24/89 - extracted from EZ_UPDATE_COMMON
*
* Arguments:
*	var		- variable number in COMMON
*
	INTEGER		var
*                                                 
* Include files
#ifdef unix
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* calling argument declarations:

* Local definitions
	INTEGER*4	dset, lcnt, line
*
* Is it an EZ data set ?
	dset = ds_var_setnum( var )
	IF (ds_type(dset) .NE. '  EZ' ) GOTO 9000
*
* Update variable records
	DO 100 lcnt = 1,4
	  line = grid_line(lcnt,ds_grid_number(var))		! temporary var
	  IF (line .EQ. mpsnorm) THEN
	    ds_grid_start(lcnt,var) = 1
	    ds_grid_end(lcnt,var)   = 1
	  ELSEIF (line .EQ. mpsunkn ) THEN
	    GOTO 9000
	  ELSE
	    ds_grid_start(lcnt,var) = 1
	    ds_grid_end(lcnt,var)   = line_dim(line)
	  ENDIF
  100	CONTINUE

* All done
	RETURN
*
* ERROR MESSAGES                   
 9000	STOP 'EZ_UPDATE_VAR'
	END                        
