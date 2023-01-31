	SUBROUTINE MC_INIT_DSET(dset_num,lunit,perm,tregular,use_strict,do_dsg,status)
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
* Initialize a multifile (cdf) dataset.
*    - read in stepfile information from descriptor
*    - call cd_init_dset on first file of data set to set up all 
*      common information for the dataset, such as variables, etc
*    - read gridfile containg time axis definition and associate the
*      time axis with the parent grids
*
* *kob* 6/95  - modified from gt_init_dset.F
* *kob*	9/96  - uncommented check for delta = 0.  This should work and should
*		indicate a file with only one timestep in it, and therefore
*		a possible irregular data set.  At this time, this is not
*		implemented, and instead a message is issued which tells
*		the user to use 1 as the delta rather than 0.  
*
* *kob* 9/96  - it was decided by steve and myself that this code can not
*		handle mc data sets of netcdf files which have irregular
*		timesteps internally.  THis is because the timesteps are
*		never read from the netcdf files, except for the first one.
*		This opens up a can of worms, and we decided it would 
*		probably be best to await the netcdf-library capability to
*		access multiple netCDF files.
* Linux port -kob - 3/97 - Modify include for tmap_dset.parm to be a
*			   preprocessor include for F90 - needed because
*			   tmap_dset.parm contains an ifdef
*
* V500 *kob* 4/99 - removed unneeded variable_record namelista
*	*sh* 7/99 - added perm argument needed for netCDF permutations
* V510 *sh* 3/00 - added tregular argumant
* v581 *acm* 4/2005  Add sf_ndxs and sf_ndxe to use indices rather than timesteps
*                     for indexing mc datasets in mc_read.
* V65  *acm* 2/10- all warnings and notes to std error not std out
* V68  *acm* 1/12- added use_strict argument for ticket 1910: micro-adjusting on 
*                  repeated coordinates in netCDF/OPeDNAP datasets
* V682 *acm* 9/12  Fix ticket 1969: stepfiles may have 3- or 4-digit extensions 
*                  when listed under S_NUM_OF_FILES
* V698+ *sh* 11/15 new arg in call to CD_INIT_DSET ... not relevant here
* V698  3/16 *acm* Removed Philander-Seigel code including diagnostic variables, 
* pre-defined axes and grids.
* V745 *acm*  5/20  added mode_dsg to control turning off all DSG ops

* arguments:
*	dset_num	- number assigned to this data set (output)
*			  ( 1 <= dset_num <= maxdsets - see COMMON XDSET_INFO)
*	lunit		- logical unit descriptor is opened on
*	perm		- permutation of data (unused for MC data sets)
*	status		- return status
*
* 
* Argument definitions
	LOGICAL tregular, use_strict,do_dsg
	INTEGER		dset_num, lunit, perm(*), status

* Parameter and common files

	include 'tmap_errors.parm'		! error codes
#include "tmap_dset.parm"	
	include 'tmap_dims.parm'		! data set dimensions
#include "gt_lib.parm"
	include 'xio.cmn_text'
        external xio_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xdset_info.cmn_text'		! data set common
	external xdset_info_data
	include 'xstep_files.cmn_text'		! step file common
	external xstep_files_data

* Define variables from Set File namelist records

	include 'descript.def'


* Internal declarations
*	CHARACTER*18	reading_nml
*	INTEGER*4	stp_pos, min_file, max_file, file_num,
*     .			cnt_stp, istat
*        INTEGER*4       upstat
*	INTEGER		TM_GET_GRIDNUM, TM_LENSTR, STR_UPCASE
*	REAL		last_delta, file_trange, steps_in_file
*	LOGICAL		TM_FPEQ

        CHARACTER*11    temp_t0time
	CHARACTER*13	TM_STRING
	CHARACTER*18	reading_nml
	CHARACTER*64    string
	INTEGER*4	var_cnt, var_pos, ivar, npos, bypoint, byline,
     .			byplane, l_ivar, stp_pos, min_file, max_file,
     .			file_num, cnt_stp, nrecperstep, epos, first_file,
     .			file_cnt, stp_cnt, min_pos, max_pos,istat, lcnt,
     .			lline, file_reclen, prev_steps, lenb, dot, ic
        INTEGER*4       upstat
	INTEGER		TM_GET_GRIDNUM, TM_LENSTR, STR_UPCASE
	REAL*8		last_delta, file_trange, steps_in_file
	LOGICAL		TM_FPEQ

* local parameter definitions
* NOTE (11/15) - this parameter exists to avoid the chance that DEF DATA/AGG/T 
* may inadvertently modify the time axis of a previously inited dataset.
* It is remotely possible that the same risk (never detected) exists with
* MC datasets.  If so, then just set this to .FALSE. to fix it.
* in the MC dataset code -- that like
	LOGICAL ok_share_taxis
	PARAMETER (ok_share_taxis = .TRUE.)

* record 3 - required message 
*
	NAMELIST /MESSAGE_RECORD/d_alert_on_open,
     .				d_alert_on_output,d_message
*
*
*
	NAMELIST /EXTRA_RECORD/ d_extra_des_info
*
	NAMELIST /STEPFILE_RECORD/s_filename,s_aux_set_num,s_start,
     .				  s_end,s_delta,s_num_of_files,
     .				  s_regvarflag,s_index


	reading_nml = 'MESSAGE_RECORD'
	READ (lunit, NML=MESSAGE_RECORD, END=5000, ERR=5100)
	ds_alert_on_open(dset_num) = d_alert_on_open
	ds_alert_on_output(dset_num) = d_alert_on_output
	ds_message(dset_num) = d_message
	IF (d_alert_on_open) THEN
	   CALL WARN ('regarding '//
     .	      ds_name(dset_num)(:TM_LENSTR(ds_name(dset_num)))
     .	      //' ...')
	   CALL WARN(d_message)
	ENDIF

* Last data set record - EXTRA_RECORD

	reading_nml = 'EXTRA_RECORD'

	READ (lunit, NML=EXTRA_RECORD, END=5000, ERR=5100)
*	ds_extra_des_info(dset_num) = d_extra_des_info

* step records - one for each step file stored

	stp_pos = 1
	min_file = maxstepfiles
	max_file = 0
	file_num = 0
*	ds_regsteps(dset_num) = .TRUE.
	last_delta = -999.
	cnt_stp = 1
* kob 1/97 remove space between STEPFILE_RECORD and I2 and bump I2 up to
*        I3.  this is needed to prevent error message when more than
*        100 stepfiles are used.
        WRITE (reading_nml(1:18),'(''STEPFILE_RECORD'',I3)') cnt_stp

* Initialize some STEPFILE variables
	s_aux_set_num = 0
	s_regvarflag  = ' '
	s_index       = ' '
	s_delta	      = 0	! *sh* 9/90

        prev_steps = 0

  170	READ (lunit, NML=STEPFILE_RECORD, END=5000, ERR=5100)

* check to see if this is last step record
	IF (s_filename .EQ. end_of_stepfiles) GOTO 220

* Check for unsupported features
*kob* uncommented check for delta = 0.  This should be allowed in order to
* indicate an irregular file or a file with one timestep in it, but this is
* not supported as of now.  9/96
	  IF (s_regvarflag .NE. ' ') CALL TM_ERRMSG
     .		(merr_notsupport, status, 'MC_INIT_DSET',
     .		 dset_num, no_stepfile,
     .		 'Descriptor requests irregular variables',
     .		 no_errstring, *9900)
	  IF (s_delta .EQ. 0) CALL TM_ERRMSG
     .		(merr_notsupport, status, 'MC_INIT_DSET',
     .		 dset_num, no_stepfile,
     .		 'Descriptor requests delta of 0, please use 1',
     .		 no_errstring, *9900)
	  IF (s_index .NE. ' ') CALL TM_ERRMSG
     .		(merr_notsupport, status, 'MC_INIT_DSET',
     .		 dset_num, no_stepfile, 
     .		 'Descriptor requests INDEX files', no_errstring, *9900)

	cnt_stp = cnt_stp + 1
	WRITE (reading_nml(1:18),'(''STEPFILE_RECORD '',I2)') cnt_stp

* calculate time range per file and check for even number per file
	file_trange = (s_end+s_delta-s_start)/s_num_of_files
	IF ( s_delta .GT. 0 ) THEN
	   steps_in_file = file_trange / s_delta 
	   IF (ABS(steps_in_file/NINT(steps_in_file)-1.0) .GT. 1.E-6) THEN
              WRITE (string,1100)
 1100         FORMAT(' WARNING - NON-INTEGER NUMBER OF STEPS PER STEPFILE'/
     .               ' INDICATED IN DESCRIPTOR - TRYING TO CONTINUE ANYWAY')
              lenb = TM_LENSTR( string )
              CALL WARN( string(:lenb) )
	   ENDIF
	ENDIF


* compute how many records in each time step (with no tuples/missing vars)
	nrecperstep= nintro_recs
	DO 180 ivar = 1,maxvars
	  IF (ds_var_setnum(ivar) .NE. dset_num ) GOTO 180
	  nrecperstep= nrecperstep+ds_nrecpervar(ivar)
  180	CONTINUE

* figure out if we should read filename for file number (read P.R.G)
	IF (s_num_of_files .GT. 1) THEN
	  epos = TM_LENSTR(s_filename)
	  dot = 0
	  do ic = epos, 1, -1
	    IF (s_filename(ic:ic) .EQ. '.') GOTO 188
	    dot = dot + 1
	  enddo
  188	  CONTINUE
          dot = epos - dot + 1
	  READ  (s_filename(dot:epos),1200) first_file
 1200	  FORMAT(I3)
	ELSE
	  first_file = file_num+1
	ENDIF

	DO 210 file_cnt = first_file, first_file+s_num_of_files-1
	  file_num = file_cnt
* Find the next storage position for stepfiles in common
	  DO 190 stp_cnt = stp_pos,maxstepfiles
	    IF (sf_setnum(stp_cnt) .EQ. set_not_open) GOTO 200
  190	  CONTINUE
* No room left for new stepfiles
	  CALL TM_ERRMSG ( merr_filim, status, 'GT_INIT_DSET',
     .			   dset_num, no_stepfile,
     .			   'MAX='//TM_STRING(DBLE(maxstepfiles)),
     .			   no_errstring, *9900)

* smallest file number has smallest timestep (for ds_1step)
  200	  IF (file_cnt .LT. min_file
     .	      .AND. s_aux_set_num .EQ. main_set) THEN
	    min_file = file_cnt
	    min_pos  = stp_cnt
	  ENDIF
* largest file number has largest timestep (for ds_lastep)
	  IF (file_cnt .GT. max_file
     .	      .AND. s_aux_set_num .EQ. main_set) THEN
	    max_file = file_cnt
	    max_pos  = stp_cnt
	  ENDIF

* store values in common
* Modified to check for time-independance of data.  If so, the 1st and last
*   time steps are set to unspecified_val8 -----> 6/91 kob
          sf_reclen(stp_cnt) = file_reclen   ! *sh* 3/91
	  sf_name  (stp_cnt) = s_filename
          upstat = STR_UPCASE(temp_t0time,ds_t0time(dset_num))  ! kob 6/91
	  IF (s_num_of_files .GT. 1) THEN
	     IF (file_cnt .LE. 999) THEN
	        WRITE (sf_name(stp_cnt)(epos-3:epos),1300) file_cnt
 1300	        FORMAT('.',I3.3)
             ELSE
	        WRITE (sf_name(stp_cnt)(dot-1:dot+3),1301) file_cnt
 1301	        FORMAT('.',I4.4)
             ENDIF

 	     sf_1step(stp_cnt) = s_start +
     .				   file_trange*(file_cnt-first_file)
	     sf_lastep(stp_cnt) = sf_1step(stp_cnt)+file_trange-s_delta

	     sf_ndxs(stp_cnt) = 1 + steps_in_file* (file_cnt-first_file)
	     sf_ndxe(stp_cnt) = sf_ndxs(stp_cnt) + steps_in_file - 1

	  ELSEIF (INDEX(temp_t0time,'INDEPENDENT') .GT. 0) THEN  !
             sf_1step(stp_cnt) =  unspecified_val8               ! kob -            
             sf_lastep(stp_cnt) = unspecified_val8               ! 6/91
             sf_ndxs(stp_cnt) =  unspecified_int4          
             sf_ndxe(stp_cnt) = unspecified_int4 
          ELSE  
	     sf_1step (stp_cnt) = s_start
	     sf_lastep(stp_cnt) = s_end
	     sf_ndxs (stp_cnt) = 1 + prev_steps
c	     sf_ndxe(stp_cnt) = steps_in_file + sf_ndxs(stp_cnt) - 1 
	     sf_ndxe(stp_cnt) =INT(steps_in_file+0.5)+sf_ndxs(stp_cnt)-1   ! rounding wierdness

	  ENDIF
          prev_steps = sf_ndxe(stp_cnt)
	  sf_delta(stp_cnt) = s_delta
	  sf_setnum(stp_cnt) = dset_num
	  sf_aux_set_num(stp_cnt) = s_aux_set_num
	  sf_nrecperstep(stp_cnt) = nrecperstep
	  sf_lunit(stp_cnt) = file_not_open
	  sf_index(stp_cnt) = s_index
	  IF (s_regvarflag .EQ. ' ') sf_regvars(stp_cnt) = .TRUE.
	  IF (last_delta .EQ. -999.) last_delta = s_delta
	  IF (s_delta .NE. 0) THEN
	    sf_regtimes(stp_cnt) = .TRUE.
	  ELSE
	    sf_regtimes(stp_cnt) = .FALSE.
	  ENDIF
* removed SNGL function from evalutating last_delta because it isn't
* a real*8   *kob* 14apr94
* v51 - put it back in - last_delta IS a real*8
	  ds_regsteps(dset_num) = ds_regsteps(dset_num) .AND.
     .				  sf_regtimes(stp_cnt) .AND.
     .		                TM_FPEQ((s_delta), (last_delta))
	  stp_pos = stp_cnt
  210	CONTINUE
	GOTO 170

* successful completion
  220	ds_1step(dset_num)  = SNGL(sf_1step(min_pos))
	ds_lastep(dset_num) = SNGL(sf_lastep(max_pos))
	IF (ds_regsteps(dset_num)) THEN
	  ds_delstep(dset_num)  = SNGL(sf_delta(min_pos))
	ELSE
	  ds_delstep(dset_num) = 0.
	ENDIF
	status = merr_ok
	

	CALL CD_INIT_DSET( dset_num, perm, tregular, use_strict,
     .			   ok_share_taxis,do_dsg,status)

	GOTO 9990


* errors
 5000	CALL TM_ERRMSG (merr_nmlerr, status, 'MC_INIT_DSET',
     .			dset_num, no_stepfile,
     .			'Didn''t find: '//reading_nml,
     .			no_errstring, *9900)

 5100	CALL TM_ERRMSG (merr_nmlerr, status, 'MC_INIT_DSET',
     .			dset_num, no_stepfile,
     .			'Error in: '//reading_nml,
     .			no_errstring, *9900)

 5200	CALL TM_ERRMSG (merr_nmlerr, status, 'MC_INIT_DSET',
     .			dset_num, no_stepfile,
     .			'Error in stepfile names '//reading_nml,
     .			no_errstring, *9900)

* get out from error
 9900	CALL TM_CLOSE_SET ( dset_num, istat )

* get out
 9990	CLOSE (UNIT = lunit)
	CALL LIB_FREE_LUN (lunit)
	RETURN
	END




