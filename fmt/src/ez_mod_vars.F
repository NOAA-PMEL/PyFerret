	SUBROUTINE EZ_Mod_Vars( dset, vpos, varid,
     .			name, title, units, grid, bad, status )

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
* V533 6/01 *sh* - added argument "name"
* V540 10/01 *sh* - when title is modified, reset titl_mod to blank
*      *acm* 10/01 increase length of ds_name
*      *acm* 10/01  remove VMS include statements
* v604 *acm* 7/07 increase length of ds_name, ds_des_name
* V616  1/09 *acm* - Fix ticket 1615, If doing SET VAR/UNITS, /TITLE, /BAD on a
*                    var from an EZ dataset, set the appropriate attributes.
* V65  *acm* 1/10  Change all netcdf2-style calls and parameter names to 
*                  netcdf3-style calls, which are also consistent with netcdf4.
* V65  *acm* 1/10  Change all netcdf2-style calls and parameter names to 
*                  netcdf3-style calls, which are also consistent with netcdf4.
* V68  *acm* 1/12 The variables should be declared REAL (conversion to double prec.)
* V701  7/16 *kms* Compare ds_name to char_init2048 instead of char_init1024

* Arguments:
*	dset	- data set number
*	vpos	- variable number within COMMON/XDSET_INFO
*	name	- variable name
*	title	- variable title
*	units	- variable units
*	grid	- grid data is on for this variable
*	bad	- bad data flag
*	status	- (output)
*
	CHARACTER*(*)	name, title, units
	INTEGER*4	dset, vpos, grid, status
	REAL		bad

	INTEGER		TM_LENSTR1, 
     .                  outtype_flag, varid, attid, attype, attlen, 
     .			attoutflag
	REAL		dummy
	CHARACTER	varattname*512, varname*128, attname*128, buf*2
*
* Include files
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
        include 'netcdf.inc'

*
* Local definitions
*
* Does data set exist?
	IF (ds_name(dset) .EQ. char_init2048) GOTO 9100
*
* Defining variable title?
	IF ( title .NE. char_init80 ) THEN
           ds_var_title( vpos ) = title

           attlen = TM_LENSTR1(title)
           attname = 'long_name'
           CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
           IF (attid .GT. 0) THEN
              CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, 
     .          attname, attype, attlen, attoutflag, status )
              CALL CD_REPLACE_ATTR (dset, varid, attname, attype, attlen, 
     .        title, dummy, status)
           ENDIF
	ENDIF
*
* Defining variable units?

	IF ( units .NE. char_init16 ) THEN
           ds_var_units( vpos ) = units

           attlen = TM_LENSTR1(units)
           attname = 'units'
           CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
           IF (attid .GT. 0) THEN
              CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, 
     .             attname, attype, attlen, attoutflag, status )
              CALL CD_REPLACE_ATTR (dset, varid, attname, attype, 
     .           attlen, units, dummy, status)
           ELSE
                 attype = NF_CHAR
                 attoutflag = 1
                 CALL CD_PUT_NEW_ATTR (dset, varid, attname, attype, 
     .              attlen, attoutflag, units, dummy, status )
           ENDIF
	ENDIF
*
* Defining variable name?
*	IF (name .NE. char_init16) ds_var_code(vpos) = name
	IF (name .NE. char_init16) 
     .      CALL string_array_modify(ds_var_code_head,
     .                               vpos, name, LEN(name))
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

	IF ( bad  .NE. real4_init  ) THEN
	   CALL PURGE_FILE_VAR( vpos ) ! remove memory-resident variables
	   CALL PURGE_ALL_UVARS	 ! delete (possibly dependent) uvars
	   ds_missing_flag(vpos) = ds_bad_flag(vpos)	      
	   ds_bad_flag    (vpos) = bad

	   attlen = 1
	   attname = 'missing_value'
	   CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
	   IF (attid .GT. 0) THEN
              CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, 
     .             attname, attype, attlen, attoutflag, status )
              CALL CD_REPLACE_ATTR (dset, varid, attname, attype, attlen, 
     .           buf, bad, status)
	   ENDIF
	   attname = '_FillValue'
	   CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
	   IF (attid .GT. 0) THEN
              CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, 
     .             attname, attype, attlen, attoutflag, status )
              CALL CD_REPLACE_ATTR (dset, varid, attname, attype, attlen, 
     .           buf, bad, status)
	   ENDIF

	status = merr_ok
	ENDIF

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
