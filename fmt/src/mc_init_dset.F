	SUBROUTINE MC_INIT_DSET ( dset_num, lunit, status)
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
* arguments:
*	dset_num	- number assigned to this data set (output)
*			  ( 1 <= dset_num <= maxdsets - see COMMON XDSET_INFO)
*	lunit		- logical unit descriptor is opened on
*	status		- return status
*
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
	include 'tmap_format/xstep_files.cmn_text'		! step file common
	external xstep_files_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'		! Error codes
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'		! Data set constants
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'		! Data set dimensions
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XIO.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'		! Data set common
	INCLUDE 'TMAP_FORMAT:XSTEP_FILES.CMN'		! Step file common
#endif

* Define variables from Set File namelist records
#ifdef unix
	include 'tmap_format/descript.def'
#else
	INCLUDE 'TMAP_FORMAT:DESCRIPT.DEF'
#endif

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
	INTEGER*4	var_cnt, var_pos, ivar, npos, bypoint, byline,
     .			byplane, l_ivar, stp_pos, min_file, max_file,
     .			file_num, cnt_stp, nrecperstep, epos, first_file,
     .			file_cnt, stp_cnt, min_pos, max_pos,istat, lcnt,
     .			lline, file_reclen
        INTEGER*4       upstat
	INTEGER		TM_GET_GRIDNUM, TM_LENSTR, STR_UPCASE
	REAL*8		last_delta, file_trange, steps_in_file
	LOGICAL		TM_FPEQ


* record 3 - required message 
*
	NAMELIST /MESSAGE_RECORD/d_alert_on_open,
     .				d_alert_on_output,d_message
*
*
	NAMELIST /VARIABLE_RECORD/d_var_code,d_var_title,
     .				d_var_titl_mod,d_var_units,
     .				d_grid_name,d_missing_flag,
     .				d_bad_flag,d_ordering,
     .				d_grid_start,d_grid_end,
     .				d_precision,d_aux_set_num

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
	   CALL TM_NOTE('regarding '//
     .	      ds_name(dset_num)(:TM_LENSTR(ds_name(dset_num)))
     .	      //' ...', tooplu)
	   CALL TM_NOTE(d_message, tooplu)
	ENDIF

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
	   IF ( ABS(steps_in_file/NINT(steps_in_file) - 1.0) .GT. 1.E-6 )
     .		WRITE (tooplu,1100)
 1100		FORMAT(' WARNING - NON-INTEGER NUMBER OF STEPS PER STEPFILE'/
     .	         ' INDICATED IN DESCRIPTOR - TRYING TO CONTINUE ANYWAY')
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
	  READ  (s_filename(epos-2:epos),1200) first_file
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
	     WRITE (sf_name(stp_cnt)(epos-3:epos),1300) file_cnt
 1300	     FORMAT('.',I3.3)
	     sf_1step(stp_cnt) = s_start +
     .				   file_trange*(file_cnt-first_file)
	     sf_lastep(stp_cnt) = sf_1step(stp_cnt)+file_trange-s_delta
	  ELSEIF (INDEX(temp_t0time,'INDEPENDENT') .GT. 0) THEN  !
             sf_1step(stp_cnt) =  unspecified_val8               ! kob -            
             sf_lastep(stp_cnt) = unspecified_val8               ! 6/91
          ELSE  
	     sf_1step (stp_cnt) = s_start
	     sf_lastep(stp_cnt) = s_end
	  ENDIF
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
	  ds_regsteps(dset_num) = ds_regsteps(dset_num) .AND.
     .				  sf_regtimes(stp_cnt) .AND.
     .		     TM_FPEQ(SNGL(s_delta), last_delta)
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
	

	CALL CD_INIT_DSET( dset_num, status)

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

* get out from error
 9900	CALL TM_CLOSE_SET ( dset_num, istat )

* get out
 9990	CLOSE (UNIT = lunit)
	CALL LIB_FREE_LUN (lunit)
	RETURN
	END




