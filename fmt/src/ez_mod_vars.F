	SUBROUTINE EZ_Mod_Vars(dset,vpos,title,units,grid,bad,status)

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
*
* A routine to fill in the "missing" gaps in the XDSET_INFO entry for an EZ_
* data set. All parameters (except dset and var) are optional, however, any
* unused parameters must be filled with char_init, int4_init, or real4_init
* depending on type.
*
* revision 0.00 - 08/14/89
* revision 0.10 - 10/25/89 - *sh* immediate COMMON update via EZ_UPDATE_VAR
* revision 0.11 - 04/25/91 *sh* Unix/RISC port: longer des name
* V530: change to xez_info.cmn requires ez_lib.parm first
* V532 5/01 *sh* - when the grid is modified must handle dynamic allocation

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
	include 'tmap_dims.parm'  
	include 'ez_lib.parm'  
#include "tmap_dset.parm"  
#include "gt_lib.parm"
	include 'xez_info.cmn_text'  
	external xez_info_data
	include 'tmap_errors.parm'
	include 'xdset_info.cmn_text'  
	external xdset_info_data
	include 'xstep_files.cmn_text'
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
	  CALL TM_USE_DYN_GRID(grid)	! allocate new as used by variable
	  CALL TM_DEALLO_DYN_GRID( ds_grid_number(vpos) ) ! deallo old
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
