	SUBROUTINE EZ_Mod_Dset (dset,title,ezform,grid,skip,order,status)
*
* A routine to fill in the "missing" gaps in the XDSET_INFO entry for an EZ_
* data set. All parameters (except dset) are optional, however, any unused
* parameters must be filled with char_init, or int4_init depending on type.
*
* revision 0.00 - 08/14/89
* revision 0.01 - 04/25/91 *sh* Unix/RISC port: longer des name
* revision 1.0	- 1/10/94 *kob* file/order upgrade to allow specification
*		  of axis permutations.

* Arguments:
*	dset	- data set number
*	title	- data set title
*	ezform	- reading format
*	grid	- grid data is on (spread over all variables)
*	skip	- number of heading records to skip
* 	order   - array specifying data order	1/94 *kob*
*	status	- (output)
*
	CHARACTER*(*)	title, ezform
	INTEGER*4	dset, grid, skip, order(4), status,idim
*
* Include files
#ifdef unix
	include 'tmap_format/tmap_dims.parm'  
#include "tmap_format/tmap_dset.parm"  
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/xez_info.cmn_text'  
	external xez_info_data
	include 'tmap_format/xdset_info.cmn_text'  
	external xdset_info_data
	include 'tmap_format/xstep_files.cmn_text'
	external xstep_files_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'  
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'  
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XEZ_INFO.CMN'  
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'  
	INCLUDE 'TMAP_FORMAT:XSTEP_FILES.CMN'
#endif
*
* Local definitions
	INTEGER*4	vcnt, scnt
*
* Does data set exist?
	IF (ds_name(dset) .EQ. char_init40) GOTO 9000
*
* Defining title?
	IF (title .NE. char_init80) ds_title(dset) = title
*
* Defining reading format?
	IF (ezform .NE. char_init16) THEN
	  ds_parm_name(1,dset) = 'FORMAT'
	  ds_parm_text(1,dset) = ezform
	ENDIF
*
* Defining grid number? (fills in ALL variables)
	IF (grid .NE. int4_init) THEN
	  DO 100 vcnt = 1,maxvars
	  IF (ds_var_setnum(vcnt) .NE. dset) GOTO 100
	  ds_grid_number(vcnt) = grid
* Fill in the data order  *kob* 1/94
	  IF (order(1) .NE. int4_init) THEN
	    DO 150 idim=1,4
	    ds_ordering(idim,vcnt) = order(idim) 
 150	  CONTINUE
	  ENDIF
	  CALL EZ_UPDATE_VAR(vcnt)
 100	  CONTINUE	  
	ENDIF

*
* Defining skip header records?
	IF (skip .NE. int4_init) THEN
	  DO 200 scnt = 1,maxstepfiles
  200	  IF (sf_setnum(scnt) .EQ. dset) sf_skip(scnt) = skip
	ENDIF
	status = merr_ok
	GOTO 9990
*
* Error messages
 9000	CALL TM_ERRMSG( merr_dsetnf, status, 'EZ_MOD_DSET',
     .			no_descfile, no_stepfile,
     .			no_errstring, no_errstring, *9990)
*
* finished
 9990	RETURN
	END
