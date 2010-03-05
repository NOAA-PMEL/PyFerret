	SUBROUTINE GT_INIT_DSET ( dset_num, lunit, status)

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
* background information.
*
* Programmer Mark Verschell (from programs by Steve Hankin)
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX/VMS
*
* revision 0.00 - 02/10/87
* revision 0.10 - 05/07/87 - added grid reads and integration_tstep
* revision 0.20 - 07/07/87 - made changes to check format info in descriptor,
*			     and check type of data, to make sure this reoutine
*			     can handel it (musically speaking)
* revision 1.00 - 09/11/87 - changes to deal with new descriptor format, also
*			     implemented version checking (this will start with
*			     version 4.00 of make_descript.) Stepfiles can be
*			     added in "groups" now, and auxiliary files are
*			     done differently, also number of records is now
*			     calculated in this routine, instead of given
*			     explicitely.
* revision 1.10 - 11/03/87 - Added initialization for some variables, completely
*			     implemented ADD_PARM, eliminated warning based on
*			     MAKE_DESCRIPT revision, DS_REGSTEPS is now the
*			     logical AND of all S_DELTA to indicate true
*			     stepfile regularity, and NAMELIST READ branches
*			     END and ERR now point to different messages.
*			     This is version 1C of descriptors
* revision 1.20 - 05/04/88 - *sh*Corrected bug in search for previously initial-
*			     ized data set by the same name as the one requested
*			     Corrected bug in check for constant # of time
*			     steps in file.  "steps_in_file"-->"file_trange"
*			     Also, return merr_no_action if a match is found.
* revision 2.00 - 06/24/88 - Removed the opening of the descriptor file, the
*			     reading of FORMAT_RECORD and BACKGROUND_RECORD, and
*			     the reading of gridfiles to the overall calling
*			     routine TM_INIT_DSET
* revision 2.01 - 11/01/88 - Added check for single line being a MODULO axis
* revision 2.02 - 11/21/88 - removed variable not_used (wasn't used)
* revision 2.03 - 01/23/89 - Moved stepfile unsupported feature check to
*			     directly after namelist read
* revision 2.10 - 02/08/89 - Removed modulo checking
* revision 2.11 - 02/15/89 - Renamed GT_CLOSE_SET to TM_CLOSE_SET
* revision 2.20 -  9/12/90 - *sh* allow external time axes (s_delta=0)
* Unix/RISC port - 3/13/91 *sh*: compute record length of files here
*                          - and upcase variable codes
* 6/91 -----> kob -Time independent data capability added 
*               sf_1step and sf_lastep are set to unspecified_val8
*               if the data is time independent.
* SUN port   - 1/30/92 kob --- moved all NAMELIST definitions to immediately
*                             after declarations.  SUN Fortran didn't like them
*                             embedded in executable code
*            -2/10/92 -kob-  multiplied file_reclen by 4 because SUN uses bytes
*                            instead of words when opening a file
* SGI port -4/8/92 -kob- SGI uses words when opening a file
*
* 3/18/93  *kob/sh* - Use tm_fpeq to test regularity of time steps
*
* IBM PORT 14apr94 *kob*: IBM very picky about argument types
* *sh* 3/2/95 - informative messages issued through TM_NOTE (for GUI)
* Linux port -kob - 3/97 - Modify include for tmap_dset.parm to be a
*			   preprocessor include for F90 - needed because
*			   tmap_dset.parm contains an ifdef
* V533 *sh* 6/01 - set data type
* V65  *acm* 2/10- all warnings and notes to std error not std out

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
	include 'tmap_errors.parm'		! error codes
#include "tmap_dset.parm"	
	include 'tmap_dims.parm'		! data set dimensions
#include "gt_lib.parm"
	include 'xio.parm'
	include 'tmap_dtypes.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xdset_info.cmn_text'		! data set common
	external xdset_info_data
	include 'xstep_files.cmn_text'		! step file common
	external xstep_files_data

* Define variables from Set File namelist records
#ifdef unix
	include 'descript.def'
#else
	INCLUDE 'TMAP_FORMAT:DESCRIPT.DEF'
#endif

* Internal declarations
* added temp_t0time, upstat for time_independent grid implementation. 6/91 kob
        CHARACTER*11    temp_t0time
	CHARACTER*13	TM_STRING
	CHARACTER*18	reading_nml
	CHARACTER*128   string
	INTEGER*4	var_cnt, var_pos, ivar, npos, bypoint, byline,
     .			byplane, l_ivar, stp_pos, min_file, max_file,
     .			file_num, cnt_stp, nrecperstep, epos, first_file,
     .			file_cnt, stp_cnt, min_pos, max_pos,istat, lcnt,
     .			lline, file_reclen, lenb
        INTEGER*4       upstat
	INTEGER		TM_GET_GRIDNUM, TM_LENSTR, STR_UPCASE
	REAL		last_delta, file_trange, steps_in_file
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
     .	      //' ...', lunit_errors)
	   CALL TM_NOTE(d_message, lunit_errors)
	ENDIF

* variable descriptor records - one per each variable in file

* Note: there will be a VARIABLE_RECORD read for each variable 
	reading_nml = 'VARIABLE_RECORD  1'
	var_cnt = 1
	var_pos = 1

* Initialize some VARIABLE variables
	d_var_titl_mod = ' '
	d_aux_set_num  = 0
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
	CALL TM_ERRMSG ( merr_varlim, status, 'GT_INIT_DSET',
     .			 dset_num, no_stepfile,
     .			 'MAX='//TM_STRING(DBLE(maxvars)),
     .			 no_errstring, *9900)

* Check for unsupported features
  120	IF (d_precision .NE. 'SINGLE') CALL TM_ERRMSG
     .		(merr_notsupport, status, 'GT_INIT_DSET',
     .		 dset_num, no_stepfile,
     .		 'CHANGE DOUBLE to SINGLE for VAR='//d_var_title,
     .		 no_errstring, *9900)

	var_pos = ivar
	var_cnt = var_cnt + 1
	WRITE (reading_nml(1:18),'(''VARIABLE_RECORD '',I2)') var_cnt

*        istat = STR_UPCASE( ds_var_code(ivar), d_var_code )     ! 3/91
        CALL string_array_modify_upcase(ds_var_code_head, ivar,
     .                                  d_var_code, LEN(d_var_code))

	ds_var_setnum(ivar) = dset_num
	ds_var_type(ivar) = ptype_float
	ds_var_units(ivar) = d_var_units
	ds_var_titl_mod(ivar) = d_var_titl_mod
	ds_var_title(ivar) = d_var_title
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
	DO 130 npos = 1,4
	  IF (d_ordering(npos) .EQ. 'WE') THEN
	    ds_ordering(npos,ivar) = 1
	  ELSE IF (d_ordering(npos) .EQ. 'EW') THEN
	    ds_ordering(npos,ivar) = -1
	  ELSE IF (d_ordering(npos) .EQ. 'SN') THEN
	    ds_ordering(npos,ivar) = 2
	  ELSE IF (d_ordering(npos) .EQ. 'NS') THEN
	    ds_ordering(npos,ivar) = -2
	  ELSE IF (d_ordering(npos) .EQ. 'UD') THEN
	    ds_ordering(npos,ivar) = 3
	  ELSE IF (d_ordering(npos) .EQ. 'DU') THEN
	    ds_ordering(npos,ivar) = -3
	  ELSE IF (d_ordering(npos) .EQ. 'TI') THEN
	    ds_ordering(npos,ivar) = 4
	  ELSE
	    ds_ordering(npos,ivar) = 0
	  ENDIF
  130	CONTINUE
	bypoint = ds_ordering(1,ivar)
	byline  = ds_ordering(2,ivar)
	byplane = ds_ordering(3,ivar)

	ds_precision(ivar) = d_precision(1:1)
	ds_aux_set_num(ivar) = d_aux_set_num

	ds_ndataperrec(ivar) = d_grid_end(bypoint) -
     .			       d_grid_start(bypoint) + 1
	ds_nrecpervar(ivar) = (d_grid_end(byline) -
     .			            d_grid_start(byline) + 1)*
     .			      (d_grid_end(byplane) -
     .			            d_grid_start(byplane) + 1)

* locate defining grid for each variable
	ds_grid_number(ivar) = TM_GET_GRIDNUM(d_grid_name)
	IF (ds_grid_number(ivar) .EQ. unspecified_int4) CALL TM_ERRMSG
     .		(merr_unkgrid, status, 'GT_INIT_DSET',
     .		 dset_num, no_stepfile,
     .		 'NAME='//d_grid_name, no_errstring, *9900)

* *sh* 3/91 Compute data file record length needed for this variable.
* Largest record needed for the variables will determine the file reclen.
        IF ( ds_ndataperrec(ivar) .GT. file_reclen )
     .                         file_reclen = ds_ndataperrec(ivar)

* Read next var_record
	GOTO 100

* compute how many records preceed each variable (with no tuples/missing vars)
 140    DO 160 ivar = 1,maxvars
	  IF (ds_var_setnum(ivar) .NE. dset_num) GOTO 160
	  DO 150 l_ivar = ivar-1,1,-1
	    IF (ds_var_setnum(l_ivar) .NE. dset_num) GOTO 150
	    IF (ds_aux_set_num(ivar) .EQ. ds_aux_set_num(l_ivar)) THEN
	      ds_nrecb4var(ivar) = ds_nrecb4var(l_ivar) +
     .	  			   ds_nrecpervar(l_ivar)
	      GOTO 160
	    ENDIF
  150	  CONTINUE
	  ds_nrecb4var(ivar) = nintro_recs
  160	CONTINUE

* *sh* 3/91 data file record length is longest variable plus 2 lead-in words
* with a minimum size of 8 longwords
        file_reclen = file_reclen + 2
* SGI port 4/8/92 kob
#ifdef sgi
      continue
#else
* changed ifdef from sun to reclen_in_bytes *kob* 4/97
#     ifdef reclen_in_bytes
           file_reclen = file_reclen*4
#     endif
#endif
        IF ( file_reclen .LT. 8 ) file_reclen = 8

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
	ds_regsteps(dset_num) = .TRUE.
	last_delta = -999.
	cnt_stp = 1
	WRITE (reading_nml(1:18),'(''STEPFILE_RECORD '',I2)') cnt_stp

* Initialize some STEPFILE variables
	s_aux_set_num = 0
	s_regvarflag  = ' '
	s_index       = ' '
	s_delta	      = 0	! *sh* 9/90

  170	READ (lunit, NML=STEPFILE_RECORD, END=5000, ERR=5100)
* check to see if this is last step record
	IF (s_filename .EQ. end_of_stepfiles) GOTO 220

* Check for unsupported features
	  IF (s_regvarflag .NE. ' ') CALL TM_ERRMSG
     .		(merr_notsupport, status, 'GT_INIT_DSET',
     .		 dset_num, no_stepfile,
     .		 'Descriptor requests irregular variables',
     .		 no_errstring, *9900)
!	  IF (s_delta .EQ. 0) CALL TM_ERRMSG
!     .		(merr_notsupport, status, 'GT_INIT_DSET',
!     .		 dset_num, no_stepfile,
!     .		 'Descriptor requests irregular timesteps',
!     .		 no_errstring, *9900)
	  IF (s_index .NE. ' ') CALL TM_ERRMSG
     .		(merr_notsupport, status, 'GT_INIT_DSET',
     .		 dset_num, no_stepfile, 
     .		 'Descriptor requests INDEX files', no_errstring, *9900)

	cnt_stp = cnt_stp + 1
	WRITE (reading_nml(1:18),'(''STEPFILE_RECORD '',I2)') cnt_stp

* calculate time range per file and check for even number per file
	file_trange = (s_end+s_delta-s_start)/s_num_of_files
	IF ( s_delta .GT. 0 ) THEN
	   steps_in_file = file_trange / s_delta 
	   IF ( ABS(steps_in_file/NINT(steps_in_file) - 1.0) .GT. 1.E-6 )
     .		WRITE (string,1100)
 1100		FORMAT(' WARNING - NON-INTEGER NUMBER OF STEPS PER STEPFILE'/
     .	         ' INDICATED IN DESCRIPTOR - TRYING TO CONTINUE ANYWAY')
	        lenb = TM_LENSTR( string )
	        CALL TM_NOTE( string(:lenb), lunit_errors )
	ENDIF

* compute how many records in each time step (with no tuples/missing vars)
	nrecperstep= nintro_recs
	DO 180 ivar = 1,maxvars
	  IF (ds_var_setnum(ivar) .NE. dset_num .OR.
     .	      ds_aux_set_num(ivar) .NE. s_aux_set_num) GOTO 180
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
  220	ds_1step(dset_num)  = sf_1step(min_pos)
	ds_lastep(dset_num) = sf_lastep(max_pos)
	IF (ds_regsteps(dset_num)) THEN
	  ds_delstep(dset_num)  = sf_delta(min_pos)
	ELSE
	  ds_delstep(dset_num) = 0.
	ENDIF
	status = merr_ok
	GOTO 9990

* errors
 5000	CALL TM_ERRMSG (merr_nmlerr, status, 'GT_INIT_DSET',
     .			dset_num, no_stepfile,
     .			'Didn''t find: '//reading_nml,
     .			no_errstring, *9900)

 5100	CALL TM_ERRMSG (merr_nmlerr, status, 'GT_INIT_DSET',
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
