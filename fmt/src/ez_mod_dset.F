	SUBROUTINE EZ_Mod_Dset (dset,title,ezform,grid,skip,order,status)
*
*
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
