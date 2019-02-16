        SUBROUTINE DEFINE_T_AGG_MC_DSET(dset_name, dset_path,
     .                   dset_title, nagfiles, memb_filename,
     .                   tline, tmatch, dset_num, status)

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
* Define and initialize a MC (T-aggregated netCDF) dataset from a list
* of member filenames and metadata about the times they contain
* Since this code is derived from the old MC descriptor file logic, its
* internals simulate the reading of a descriptor file
* The first netCDF file in the list will be opened and inspected
* in the process of creating the aggregation.

* V695+ *sh* 10/15 new routine -- built as hack on tm_init_dset.F
* V698 *sh* 1/16  pass explicit path name (for "USE file.agg" compatibility)
*                 date-sort the file list in BUILD_T_AGG_AXIS

* arguments:

*	dset_name	- dataset name to assign to the aggregation (input)
*	dset_path	- dataset path to assign to the aggregation (input)
*	dset_title	- dataset title (input)
*	nagfiles	- number of files in the aggregation (input)
*	memb_filename	- array of filenames (input)
*       tline           - user-provided time axis to use (input, can be unspecified)
*       tmatch          - method of matching times to tline (input, can be empty);
*                         eg, '@ASN', '@BIN', '@XACT', '@XACT:0.001'
*	dset_num	- number assigned to this data set (output)
*	status		- return status (output)
*

        IMPLICIT NONE
* Argument definitions
	INTEGER		nagfiles, dset_num, tline, status
	CHARACTER*(*)	dset_name, dset_path, dset_title, tmatch
	CHARACTER*512	memb_filename(nagfiles)

* Parameter and common files
        include 'ferret.parm'
        include 'errmsg.parm'
        include 'tmap_errors.parm'              ! error codes
#include "tmap_dset.parm"
        include 'tmap_dims.parm'                ! data set dimensions
	include 'xtm_grid.cmn_text'
	include 'xio.cmn_text'
        external xio_data
        include 'xdset_info.cmn_text'   ! data set common
        external xdset_info_data
        include 'xsys_names.cmn_text'
        external xsys_names_data
        include 'xdiag_ctrl.cmn_text'
        external xdiag_ctrl_data
        include 'xrisc_buff.cmn'

* Internal declarations
        INTEGER memb_nstep_start(nagfiles), memb_nsteps(nagfiles),
     .          member_order(nagfiles)
	REAL*8  memb_tstep_start(nagfiles), memb_tstep_end(nagfiles)
        LOGICAL	match
	INTEGER	TM_LENSTR1, tmap_status
	INTEGER	dlen, slen, iset, idim
	INTEGER	perm(nferdims), memb_order(nagfiles)
	CHARACTER LEFINT*16, buff*16
        LOGICAL ok_re_use_existing, tregular, use_strict

* initialize
	dlen = TM_LENSTR1( dset_name )
	DO idim = 1, nferdims
 	   perm(idim) = unspecified_int4 ! dummy USE/ORDER= qualifier
	ENDDO

* check to see if the aggregation name is already in use
	DO iset = 1, maxdsets
           match = ds_name(iset) .EQ. dset_name
           IF ( match ) GOTO 5100
  	ENDDO

* initialize the first file in the aggregation
* ... it will serve as a pattern of the time-dependent variables that will
*     be presumed to exist in all other files of the aggregation
        ok_re_use_existing = .FALSE.
        tregular = .FALSE.
        use_strict = .FALSE.
        CALL TM_INIT_DSET(memb_filename(1), ok_re_use_existing, perm,
     .                    dset_num, tregular, use_strict, tmap_status)

	IF (tmap_status.NE.merr_ok) GOTO 5010
	CALL CD_NCCLOSE ( dset_num, status)
	IF ( status .NE. merr_ok ) GOTO 5010

        IF ( tline .NE. unspecified_int4 ) THEN
* Verify and assign the given time axis for the aggregration
           CALL ASSIGN_T_AGG_AXIS(tline, tmatch, dset_num, nagfiles,
     .             memb_filename, memb_tstep_start, memb_tstep_end,
     .             memb_nstep_start, memb_nsteps, memb_order, status)
        ELSE
* Build the aggregated time axis by reading the coord from the netCDF files
           CALL BUILD_T_AGG_AXIS(dset_num, nagfiles, memb_filename,
     .             memb_tstep_start, memb_tstep_end, memb_nstep_start,
     .             memb_nsteps, memb_order, status)
        ENDIF
        IF (status .NE. ferr_ok) GOTO 9900

* Use individual file metadata to simulate an MC descriptor ("Multifile Cdf")
        CALL INIT_T_AGG_MC_DSET(dset_num, nagfiles, memb_filename,
     .          memb_tstep_start, memb_tstep_end, memb_nstep_start,
     .          memb_nsteps, memb_order, status)
	IF (status .NE. merr_ok) GOTO 9900

* change the dataset metadata to make this into an aggregation
	ds_type(dset_num) = '  MC'
	ds_name(dset_num) = dset_name
	ds_des_name(dset_num) = dset_path
	IF (dset_title .NE. char_init) ds_title(dset_num) = dset_title
	buff = LEFINT(nagfiles, slen)
	ds_mod_title(dset_num)      = 'T-aggregation of '
     .				// buff(:slen)//' netCDF files'
	ds_regsteps(dset_num) = line_regular(ds_time_axis(dset_num))

! stuff done in MC descriptors.  Needed here ??
!	CALL TM_MAKE_4D_GRIDS(dset_num,status)
!* ?? need to do this for remote mc aggregations ??
!	CALL TM_DSET_USE_GRIDS( dset_num )

* success
	RETURN

* errors
 5010	CALL ERRMSG( ferr_TMAP_error, status, ' ', *5000 )

 5100   CALL ERRMSG( ferr_invalid_command,status,
     .		'given aggregation dataset name already in use: '
     .		//dset_name(:dlen), *5000 )

* get out from error
 9900	CALL TM_PURGE_TMP_GRIDS(TMAP_status)
	CALL TM_CLOSE_SET ( dset_num, TMAP_status )
 9950	dset_num = unspecified_int4
 5000	RETURN
	END
