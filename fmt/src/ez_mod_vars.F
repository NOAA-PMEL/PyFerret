	SUBROUTINE EZ_Mod_Vars(dset,vpos,title,units,grid,bad,status)
*
* A routine to fill in the "missing" gaps in the XDSET_INFO entry for an EZ_
* data set. All parameters (except dset and var) are optional, however, any
* unused parameters must be filled with char_init, int4_init, or real4_init
* depending on type.
*
* revision 0.00 - 08/14/89
* revision 0.10 - 10/25/89 - *sh* immediate COMMON update via EZ_UPDATE_VAR
* revision 0.11 - 04/25/91 *sh* Unix/RISC port: longer des name
* Arguments:
*	dset	- data set number
*	vpos	- variable number within COMMON/XDSET_INFO
*	title	- variable title
*	units	- variable units
*	grid	- grid data is on for this variable
*	bad	- bad data flag
*	status	- (output)
*
	CHARACTER*(*)	title, units
	INTEGER*4	dset, vpos, grid, status
	REAL*4		bad
*
* Include files
#ifdef unix
	include 'tmap_format/tmap_dims.parm'  
#include "tmap_format/tmap_dset.parm"  
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xez_info.cmn_text'  
	external xez_info_data
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/xdset_info.cmn_text'  
	external xdset_info_data
	include 'tmap_format/xstep_files.cmn_text'
	external xstep_files_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'  
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'  
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XEZ_INFO.CMN'  
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'  
	INCLUDE 'TMAP_FORMAT:XSTEP_FILES.CMN'
#endif
*
* Local definitions
*
* Does data set exist?
	IF (ds_name(dset) .EQ. char_init40) GOTO 9100
*
* Defining variable title?
  200	IF (title .NE. char_init80) ds_var_title(vpos) = title
*
* Defining variable units?
	IF (units .NE. char_init16) ds_var_units(vpos) = units
*
* Defining grid number?
	IF (grid .NE. int4_init) THEN
	  ds_grid_number(vpos) = grid
	  CALL EZ_UPDATE_VAR(vpos)
	ENDIF
*
* Defining bad data flag?
	IF (bad .NE. real4_init) ds_bad_flag(vpos) = bad
	status = merr_ok
	GOTO 9990
*
* Error messages
 9100	CALL TM_ERRMSG( merr_dsetnf, status, 'EZ_MOD_DSET',
     .			no_descfile, no_stepfile,
     .			no_errstring, no_errstring, *9990)
*
* finished
 9990	RETURN
	END
