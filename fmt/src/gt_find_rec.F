	SUBROUTINE GT_FIND_REC ( sf_num,time_step,varpos,varstrt,status )

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
* in the indicated file
* this routine assumes that the file has already been OPENed via GT_OPEN_STEP

* programmer - mark verschell
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system 
*
* revision 0.00 - 11/19/85 - (steve hankin)
* revision 1.00 - 02/14/87 - changed routine to locate variable position not
*			     step position (mav)
* revision 1.01 - 09/19/88 - changed timestep comparison to single precision
* revision 1.10 - 09/12/90 - *sh* added provision for irregular time axis
* Unix/RISC port - 3/91 *sh*: added logic to convert cpu binary representation
* 6/91 ----> kob - Time independent data handling capability put in
*                  Had to include file tmap_format/gt_lib.parm (as in 
*                  gt_open_step.f) in order to have unspecified_val8
*                  declared.  
* SUN port 2/92 - machine type word conversions

* arguments:
*	sf_num		-  file # within COMMON/XSTEP_FILES/
*	time_step	-  desired time step
*	varpos		-  position of variable in data set common
*	vastrt		-  (output) record number of the first variable record
*	status		-  INTEGER, output: error status at completion

* calling argument declarations:
	INTEGER		sf_num,status,varpos,varstrt
	REAL*8		time_step

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/tmap_errors.parm'	! error code parameter defs
        include 'tmap_format/gt_lib.parm'       ! put in 6/91 for time ind. kob
	include 'tmap_format/xstep_files.cmn_text'
	external xstep_files_data
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
* xmachine_byte.cmn on sun becauses it uses BYTE where ULTRIX uses INTEGER*1
#ifdef sun
      include 'tmap_format/xmachine_byte.cmn'
#else
      include 'tmap_format/xmachine_int1.cmn'
#endif
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'	! error code PARAMETER defs
	INCLUDE 'TMAP_FORMAT:XSTEP_FILES.CMN'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XMACHINE_BYTE.CMN'
#endif

* internal variable declarations:
	INTEGER*2	i2dummy,nvarsinstep
	INTEGER		setnum,nrecsinstep,lunit,nstep,step_rec,
     .			sequence_number, grid, cptype
	REAL*8		file_tstep
	CHARACTER*4	rec_code,run_num,mod_num
	INTEGER		TM_SUBSC
	CHARACTER*13	TM_STRING
	LOGICAL		TM_FPEQ

* internal parameter declarations:
* commented out the declaration of t_dim because with the
* inclusion of tmap_format/gt_lib.parm, it was declared already.  kob 6/91
*	INTEGER	t_dim
*	PARAMETER ( t_dim = 4 )

* COMMON to save clues for efficiency
	INTEGER l_sf1, l_tstep
	COMMON / XFREC_SAVE / l_sf1, l_tstep

* initialize
	setnum = sf_setnum(sf_num)		! internal data set number
	lunit = sf_lunit(sf_num)		! unit # for file
	varstrt = 0.0				! in case the time isn't here
	cptype = sf_cpu_type(sf_num)

* number of preceeding time steps
!	IF ( .NOT.sf_regvars(sf_num) )
!     .	  CALL TM_ERRMSG(merr_notsupport, status, 'GT_FIND_REC',
!     .			 setnum, sf_num,
!     .			 'Irregular variables or timesteps,
!     .			 no_errstring, *5000)
	IF ( sf_regtimes(sf_num)) THEN
	   nstep = ((time_step - sf_1step(sf_num)) / sf_delta(sf_num))
     .			*1.000001
	ELSE	! rev 1.10 additions
* ... (previous subscript values to provide performance-enhancing "clues")
	   grid = ds_grid_number(varpos)
	   l_sf1   = TM_SUBSC ( sf_1step(sf_num), grid, t_dim, l_sf1 )
	   l_tstep = TM_SUBSC ( time_step,        grid, t_dim, l_tstep+1 )
	   nstep = l_tstep - l_sf1
	ENDIF

* compute record number of 'STEP' record at start of data for desired time step
	step_rec = sf_skip(sf_num)		 ! leading file records to skip
     .		  + nstep*sf_nrecperstep(sf_num) ! preceeding time steps
     .		  + 1				 ! 'STEP' record

* check for correct STEP record at computed position
	READ (UNIT=lunit, REC=step_rec, ERR=6000)rec_code,
     .						file_tstep,
     .						nrecsinstep,
     .						nvarsinstep,
     .						i2dummy,
     .						run_num,
     .						mod_num

* file written on a different machine ? convert machine words
	IF ( cptype .NE. (active_cpu) ) THEN
	   CALL TM_INT4_SWAP( nrecsinstep, cptype )
	   CALL TM_INT2_SWAP( nvarsinstep, cptype )
	   CALL TM_DFP_CONVERT( file_tstep, cptype )
	ENDIF

* is it a 'STEP' record  
	IF (rec_code .NE. 'STEP') CALL TM_ERRMSG
     .		( merr_filpos, status, 'GT_FIND_REC', setnum, sf_num,
     .		  'STEP'//' TSTEP='//TM_STRING(time_step),
     .		  rec_code//' TSTEP='//TM_STRING(file_tstep), *5000)

* is the time step value correct ?
* If time step equals unspecified_val8 it is okay because this
* is the flag for time independent data
	IF ( .NOT. TM_FPEQ(SNGL(file_tstep), SNGL(time_step)) 
     .    .AND. (time_step .NE. unspecified_val8))
     .	     CALL TM_ERRMSG ( merr_tstep, status, 'GT_FIND_REC', setnum,
     .	                      sf_num, TM_STRING(time_step),
     .	                      TM_STRING(file_tstep), *5000 )

* is it from the correct run ?
	IF ( run_num .NE. ds_expnum(setnum) ) CALL TM_ERRMSG
     .		( merr_expnum, status, 'GT_FIND_REC', setnum, sf_num,
     .		  ds_expnum(setnum), run_num, *5000)

* does it have the correct run modification code (warning only, if not) ?
	IF (sf_aux_set_num(sf_num) .EQ. main_set) THEN
* main data set
	  IF (mod_num .NE. ds_modnum(setnum) ) CALL TM_ERRMSG
     .		( merr_modnum, status, 'GT_FIND_REC', setnum, sf_num, 
     .		  ds_modnum(setnum), mod_num, *2000)
	ELSE
* auxiliary data set so locate correct entry
	  IF (mod_num .NE. ds_aux_modnum(sf_aux_set_num(sf_num),setnum))
     .	   CALL TM_ERRMSG ( merr_modnum, status, 'GT_FIND_REC',
     .			   setnum, sf_num,
     .			   ds_aux_modnum(sf_aux_set_num(sf_num),setnum),
     .			   mod_num, *2000)
	ENDIF

* compute position of start of variable record
2000	varstrt = step_rec + ds_nrecb4var(varpos)

* check for correct variable record at computed position
	READ (UNIT=lunit, REC=varstrt, ERR=6000)rec_code,
     .						sequence_number

* file written on a different machine ? convert sequence number
	IF ( cptype .NE. (active_cpu) ) THEN
	   CALL TM_INT4_SWAP( sequence_number, cptype )
	ENDIF

* is it the correct variable
	IF (rec_code .NE. ds_var_code(varpos)) CALL TM_ERRMSG
     .		( merr_filpos, status, 'GT_FIND_REC', setnum, sf_num,
     .		  ds_var_code(varpos)//' TSTEP='//TM_STRING(time_step),
     .		  rec_code//' TSTEP='//TM_STRING(file_tstep), *5000)

* got it
 	status = merr_ok
 5000	RETURN

* ERR= exit
 6000	CALL TM_ERRMSG (merr_erreq, status, 'GT_FIND_REC',
     .			setnum, sf_num,
     .			no_errstring, no_errstring, *5000)
	END
