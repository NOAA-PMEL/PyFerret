	SUBROUTINE TS_FIND_VAR ( dset_num,varpos,vf_num,varstrt,status )
*
* Locates and opens (if not already open) the correct data file containing
* the specified variable.
* 
* programmer - mark verschell
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 08/23/88
*
* arguments:
*	dset_num	- number (in COMMON/DSET_INFO/) of dataset
*	varpos		- location in ds_ arrays of the correct variable
*	vf_num	 	- (output) the desired file number (in /XSTEP_FILES/)
*	varstrt		- (output) record number of the first variable record
*	status		- returned error code

	INTEGER		dset_num,vf_num,varstrt,varpos,status

#ifdef unix
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/xstep_files.cmn_text'
	external xstep_files_data
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XSTEP_FILES.CMN'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
#endif

	CHARACTER*8	variable
	INTEGER		vfcnt,ivar

	variable = ds_var_code(varpos)
* find correct file
	DO 300 vfcnt = 1,maxvarfiles
	  IF (vf_setnum(vfcnt) .NE. dset_num) GOTO 300	! right dataset?
	  IF ((vf_firstvar(vfcnt).GT. varpos) .OR.	! contains variable
     .	      (vf_lastvar(vfcnt) .LT. varpos)) GOTO 300
	  GOTO 400
  300	CONTINUE
	GOTO 9000

* found the correct one!
* check to see if file open already
  400	IF (vf_lunit(vfcnt)  .EQ. file_not_open) THEN
* let's try to open it
	  CALL TS_OPEN_VARF (vfcnt,status)
* Is it mounted?
	  IF (status .NE. merr_OK) GOTO 9990
	ENDIF

* Varfile mounted and normal so locate varstrt
	varstrt = 1 + ds_nrecb4var(varpos)

* got it
	vf_num = vfcnt
	status = merr_ok
	GOTO 9990

* ERRORS
 9000	CALL TM_ERRMSG (merr_unkvar, status, 'TS_FIND_VAR',
     .			dset_num, no_stepfile,
     .			'VAR='//variable, no_errstring, *9990)

 9990	RETURN
	END
