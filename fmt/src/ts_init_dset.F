	SUBROUTINE TS_INIT_DSET ( dset_num, lunit, status)
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
*
* Loads common blocks XDSET_INFO and XSTEP_FILES with vital system and
* background information for time series files.
*
* Programmer Mark Verschell (from programs by Steve Hankin)
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX/VMS
*
* revision 0.00 - 07/22/88 - From GT_INIT_DSET version 2.00
* revision 0.01 - 11/21/88 - Removed variable not_used (wasn't used)
* revision 0.10 - 02/08/89 - Removed modulo checking
* revision 0.11 - 02/15/89 - Changed GT_CLOSE_SET to TM_CLOSE_SET
* revision 0.12 - 01/22/90 - *sh* disabled logic to produce a time axis with
*		             range encompassinig all previous TS sets.  Logic
*			     is flawed since it is based on MIN/MAX of tsteps
*			     rather than dates
* revision 0.13 - 02/23/90 - *sh* corrected bug in ds_npospervar calculation
* Unix/RISC port - 3/19/91 *sh*: compute record length of files here
* SUN port   - 1/30/92 kob --- moved all NAMELIST definitions to immediately
*                             after declarations.  SUN Fortran didn't like them
*                             embedded in executable code
* Linux port -kob - 3/97 - Modify include for tmap_dset.parm to be a
*			   preprocessor include for F90 - needed because
*			   tmap_dset.parm contains an ifdef

* arguments:
*	dset_num	- number assigned to this data set (output)
*			  ( 1 <= dset_num <= maxdsets - see COMMON XDSET_INFO)
*	lunit		- logical unit descriptor is opened on
*	status		- return status
*

* Argument definitions
	INTEGER		dset_num, lunit, status

* Parameter and common files
#ifdef unix
	include 'tmap_format/tmap_errors.parm'		! error codes
#include "tmap_format/tmap_dset.parm"	
	include 'tmap_format/tmap_dims.parm'		! data set dimensions
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xio.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'tmap_format/xdset_info.cmn_text'		! data set common
	external xdset_info_data
	include 'tmap_format/xstep_files.cmn_text'		! var file common
	external xstep_files_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'		! Error codes
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'		! Data set constants
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'		! Data set dimensions
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XIO.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'		! Data set common
	INCLUDE 'TMAP_FORMAT:XSTEP_FILES.CMN'		! Var file common
#endif

* Define variables from Set File namelist records
#ifdef unix
	include 'tmap_format/descript.def'
#else
	INCLUDE 'TMAP_FORMAT:DESCRIPT.DEF'
#endif

* Internal declarations
	CHARACTER*13	TM_STRING
	CHARACTER*18	reading_nml
	INTEGER*4	var_cnt, var_pos, ivar, npos, varf_pos, cnt_varf,
     .			varf_cnt, istat, lvar, lcnt, lline, file_reclen
	INTEGER		TM_GET_GRIDNUM, TM_LENSTR, STR_UPCASE

* record 3 - required message 

	NAMELIST /MESSAGE_RECORD/d_alert_on_open,
     .				d_alert_on_output,d_message

* variable descriptor records - one per each variable in file

	NAMELIST /VARIABLE_RECORD/d_var_code,d_var_title,
     .				d_var_titl_mod,d_var_units,
     .				d_grid_name,d_missing_flag,
     .				d_bad_flag,d_grid_start,d_grid_end,
     .				d_var_tstart,d_var_tend,
     .				d_var_tdelta,d_ndataperrec,
     .				d_precision

*
	NAMELIST /EXTRA_RECORD/ d_extra_des_info

* varfile records - one for each varfile stored
	NAMELIST /VARFILE_RECORD/v_filename,v_firstvar,v_lastvar,v_index

	reading_nml = 'MESSAGE_RECORD'
	READ (lunit, NML=MESSAGE_RECORD, END=5000, ERR=5100)
	ds_alert_on_open(dset_num) = d_alert_on_open
	ds_alert_on_output(dset_num) = d_alert_on_output
	ds_message(dset_num) = d_message
	IF (d_alert_on_open) WRITE (tooplu,1000)
     .	      ds_name(dset_num)(:TM_LENSTR(ds_name(dset_num))),d_message
 1000	FORMAT(/' *** MESSAGE REGARDING ',A,' :'/1X,A/)


* Note: there will be a VARIABLE_RECORD read for each variable 
	reading_nml = 'VARIABLE_RECORD  1'
	var_cnt = 1
	var_pos = 1
* Initialize some VARIABLE variables
	d_var_titl_mod = ' '
        file_reclen = 0      ! *sh* 3/91

  100	READ  (lunit, NML=VARIABLE_RECORD, END=5000, ERR=5100)
* Check to see if this is the end of the variable records
	IF (d_var_code .EQ. end_of_variables) GOTO 140

* Find the next storage position for variables in VARIABLE_RECORD namelist
* variables
	DO 110 ivar = var_pos,maxvars
	  IF (ds_var_setnum(ivar) .EQ. set_not_open) GOTO 120
  110	CONTINUE
* No room left for new variables
	CALL TM_ERRMSG ( merr_varlim, status, 'TS_INIT_DSET',
     .			 dset_num, no_varfile,
     .			 'MAX='//TM_STRING(DBLE(maxvars)),
     .			 no_errstring, *9900)

* Check for unsupported features
  120	IF (d_precision .NE. 'SINGLE') CALL TM_ERRMSG
     .		(merr_notsupport, status, 'TS_INIT_DSET',
     .		 dset_num, no_varfile,
     .		 'CHANGE DOUBLE to SINGLE for VAR='//d_var_title,
     .		 no_errstring, *9900)

	ds_var_setnum(ivar) = dset_num
        istat = STR_UPCASE( ds_var_code(ivar), d_var_code )     ! 4/91
	ds_var_title(ivar) = d_var_title
	ds_var_titl_mod(ivar) = d_var_titl_mod
	ds_var_units(ivar) = d_var_units
	ds_missing_flag(ivar) = d_missing_flag
	ds_bad_flag(ivar) = d_bad_flag
	ds_grid_start(1,ivar) = d_grid_start(1)
	ds_grid_start(2,ivar) = d_grid_start(2)
	ds_grid_start(3,ivar) = d_grid_start(3)
	ds_grid_start(4,ivar) = d_grid_start(4)
	ds_grid_end(1,ivar) = d_grid_end(1)
	ds_grid_end(2,ivar) = d_grid_end(2)
	ds_grid_end(3,ivar) = d_grid_end(3)
	ds_grid_end(4,ivar) = d_grid_end(4)
	ds_precision(ivar) = d_precision(1:1)
	ds_ndataperrec(ivar) = d_ndataperrec
*  old, flawed code to produce time axis encompassing all previous:*sh* 1/90
!	IF (min_ds_tstart .GT. d_var_tstart .OR.
!     .	    min_ds_tstart .EQ. int4_init) min_ds_tstart = d_var_tstart
!	IF (max_ds_tend .LT. d_var_tend)  max_ds_tend = d_var_tend
!	IF (ds_tdelta .LT. d_var_tdelta)  ds_tdelta = d_var_tdelta
* new code to kludge around it:*sh*1/90
	min_ds_tstart = d_var_tstart
	max_ds_tend = d_var_tend
	ds_tdelta = d_var_tdelta
* end of changes:*sh* 1/90
*  old, flawed code to calculate ds_npospervar:*sh* 2/90
!	ds_npospervar(ivar) =
!     .	                 (d_grid_end(4)-d_grid_start(4)+1)/d_ndataperrec
* new code:*sh*2/90
	ds_npospervar(ivar) =
     .			(d_grid_end(4)-d_grid_start(4))/d_ndataperrec + 1
* end of changes:*sh* 2/90
	ds_nrecpervar(ivar) = ds_npospervar(ivar) *
     .	                         (d_grid_end(1)-d_grid_start(1) + 1) *
     .	                         (d_grid_end(2)-d_grid_start(2) + 1) *
     .	                         (d_grid_end(3)-d_grid_start(3) + 1) + 1

*sh* 3/91 Compute data file record length needed for this variable.
* Largest record needed for the variables will determine the file reclen.
        IF ( d_ndataperrec .GT. file_reclen )
     .                          file_reclen = d_ndataperrec

* locate defining grid for each variable
	ds_grid_number(ivar) = TM_GET_GRIDNUM(d_grid_name)
	IF (ds_grid_number(ivar) .EQ. unspecified_int4) CALL TM_ERRMSG
     .		(merr_unkgrid, status, 'TS_INIT_DSET',
     .		 dset_num, no_varfile,
     .		 'NAME='//d_grid_name, no_errstring, *9900)

* Set up to read next variable
	var_pos = ivar
	var_cnt = var_cnt + 1
	WRITE (reading_nml(1:18),'(''VARIABLE_RECORD '',I2)') var_cnt
	GOTO 100

* *sh* 3/91 data file record length is longest variable plus 5 lead-in words
* with a minimum size of 8 longwords
  140   file_reclen = file_reclen + 5
        IF ( file_reclen .LT. 8 ) file_reclen = 8
*if on a sun, file_reclen should be in bytes, not words   kob 2/92
*if on an sgi, it should be in bytes
#ifdef sgi
      continue
#else
#   ifdef sun
        file_reclen = file_reclen*4
#   endif
#endif

* Last data set record - EXTRA_RECORD

	reading_nml = 'EXTRA_RECORD'

	READ (lunit, NML=EXTRA_RECORD, END=5000, ERR=5100)
*	ds_extra_des_info(dset_num) = d_extra_des_info

* initialize incidental variables
	ds_basic_axes(1,dset_num) = mpsxt
	ds_basic_axes(2,dset_num) = mpsxu
	ds_basic_axes(3,dset_num) = mpsyt
	ds_basic_axes(4,dset_num) = mpsyu
	ds_basic_axes(5,dset_num) = mpszt
	ds_basic_axes(6,dset_num) = mpszw

* varfile records - one for each varfile stored

	varf_pos = 1
	cnt_varf = 1
	WRITE (reading_nml(1:18),'(''VARFILE_RECORD '',I2)') cnt_varf

* Initialize some VARFILE variables
	v_index       = ' '

  150	READ (lunit, NML=VARFILE_RECORD, END=5000, ERR=5100)
* check to see if this is last varfile record
	IF (v_filename .EQ. end_of_varfiles) GOTO 200

* Find the next storage position for varfiles in common
	DO 160 varf_cnt = varf_pos, maxvarfiles
	  IF (vf_setnum(varf_cnt) .EQ. set_not_open) GOTO 170
  160	CONTINUE
* No room left for new varfiles
	CALL TM_ERRMSG ( merr_filim, status, 'TS_INIT_DSET',
     .			   dset_num, no_varfile,
     .			   'MAX='//TM_STRING(DBLE(maxvarfiles)),
     .			   no_errstring, *9900)

* store values in common
  170	vf_name(varf_cnt) = v_filename
        vf_reclen(varf_cnt) = file_reclen   ! *sh* 3/91
	vf_setnum(varf_cnt) = dset_num
	vf_lunit(varf_cnt) = file_not_open
	vf_index(varf_cnt) = v_index
* Replace variable codes in firstvar/lastvar with pointers into XDSET_INFO
	vf_firstvar(varf_cnt) = int4_init
	vf_lastvar(varf_cnt) = int4_init
	DO 180 ivar = 1,maxvars
	  IF (ds_var_setnum(ivar) .EQ. dset_num) THEN
	    IF (ds_var_code(ivar) .EQ. v_firstvar)
     .        vf_firstvar(varf_cnt) = ivar
	    IF (ds_var_code(ivar) .EQ. v_lastvar)
     .        vf_lastvar(varf_cnt) = ivar
	  ENDIF
  180	CONTINUE
* Variable not declared/wrong order errors
	IF (vf_firstvar(varf_cnt) .EQ. int4_init) CALL TM_ERRMSG
     .	   (merr_unkvar, status, 'TS_INIT_DSET',dset_num, varf_cnt, 
     .	    v_firstvar//' in VARFILE_RECORD but not in VARIABLE_RECORD',
     .	    no_errstring, *9900)
	IF (vf_lastvar(varf_cnt) .EQ. int4_init) CALL TM_ERRMSG
     .	   (merr_unkvar, status, 'TS_INIT_DSET',dset_num, varf_cnt, 
     .	    v_lastvar//' in VARFILE_RECORD but not in VARIABLE_RECORD',
     .	    no_errstring, *9900)
	IF (vf_lastvar(varf_cnt) .LT. vf_firstvar(varf_cnt))
     .	    CALL TM_ERRMSG (merr_notsupport, status, 'TS_INIT_DSET',
     .		        dset_num, varf_cnt, 
     .		        'Reverse variable order in '//reading_nml(1:18),
     .	                no_errstring, *9900)

* determine number of records before variable in VAR_FILE
	lvar = vf_firstvar(varf_cnt)
	ds_nrecb4var(lvar) = 0
	DO 190 ivar = vf_firstvar(varf_cnt)+1,vf_lastvar(varf_cnt)
	  IF (ds_var_setnum(ivar) .EQ. dset_num) THEN
	    ds_nrecb4var(ivar) = ds_nrecb4var(lvar) + ds_nrecpervar(lvar)
	    lvar = ivar
	  ENDIF
  190	CONTINUE

* Check for unsupported features
	IF (v_index .NE. ' ') CALL TM_ERRMSG
     .		(merr_notsupport, status, 'TS_INIT_DSET',
     .		 dset_num, varf_cnt, 
     .		 'Descriptor requests INDEX files', no_errstring, *9900)

	varf_pos = varf_cnt
	cnt_varf = cnt_varf + 1
	WRITE (reading_nml(1:18),'(''VARFILE_RECORD '',I2)') cnt_varf
	GOTO 150

* successful completion
  200	status = merr_ok
	GOTO 9990

* errors
 5000	CALL TM_ERRMSG (merr_nmlerr, status, 'TS_INIT_DSET',
     .			dset_num, no_varfile,
     .			'Didn''t find: '//reading_nml,
     .			no_errstring, *9900)

 5100	CALL TM_ERRMSG (merr_nmlerr, status, 'TS_INIT_DSET',
     .			dset_num, no_varfile,
     .			'Error in: '//reading_nml,
     .			no_errstring, *9900)

* get out from error
 9900	CALL TM_CLOSE_SET ( dset_num, istat )

* get out
 9990	CLOSE (UNIT = lunit)
	CALL LIB_FREE_LUN (lunit)
	RETURN
	END
