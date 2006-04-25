	INTEGER*4 FUNCTION GT_GET_TIME_AXIS ( dset_num, status )
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
*  Will use the information stored in XSTEP_FILES.CMN to construct a temporary
*  time axis in XTM_GRID.CMN which will then be compared to pre-existing grids
*  stored in common. If a match is found the temporary axis is deleted and the
*  line number for the pre-existing axis is returned. If no match is found,
*  the temporary axis is moved to permanent storage, and it's line number is
*  returned.
*
*  Written 06/88 for PMEL/TMAP by M. Verschell
*
* revision 0.00 - 06/30/88
* revision 0.01 - 11/21/88 - variable unused check is now a specific char_initXX
* revision 0.10 - 12/12/88 - changed line slot location to last to first search,
*			     free line memory slot gotten by next_line_mem_pos
* revision 1.00 - 01/10/88 - Checks regularity of line before creating time axis
*			     changed check of regularity to check to see if all
*			     sf_delta's are equal. eliminated unused variables,
*			     restructured code arund these changes. Line slot
*			     now gotten by TM_FIND_LIKE_LINE.
* revision 1.01 - 02/06/88 - line_modulo not initialized - corrected *sh*
* revision 1.02 - 02/17/89 - Fixed next_line_mem_pos error(1 less than correct)
* 6/24/92 *sh* & *ko* - bug fix: logic for line_regular if gap in step files
*                     - fix to bug fix (8/92 *sh*) - prev_lastep was INTEGER 
* 8/28/96 *kob*	      - bug fix: change type of "itt" to real*8
*		      - bug fix: get rid of all bpos stuff and instead
*				 call tm_irregax_mid to set up box
*				 position stuff
* 9/5/96 *kob*	      - bug fix: change of itt to real created a problem at
*			         loop 110, because itt was being used as the
*			         loop variable.  Not all of the timesteps in
*				 each stepfile were being loaded into the 
*				 line_mem array.  Loop now based on calculated
*				 nteps_in_file.
* 5/14/96 *kob*	       -Linux port - added an "INT" in calculating nsteps in the 
*				     stepfile.  roundoff /dp differences were causing
*				     a problem in linux/nag f90, so we added a small
*				     number (dp_epsilon = 1.d-10) to make sure INT operation
*			   	     would give the correct integer.
* 12/00  *acm*		add cal_id for alternative calendar axes
* V541  4/02 *acm*      Improved error message if T0 missing or bad
* V581 *acm* 6/05 - For fix to bug 1271, add flag line_shift_origin
*            and restore original t0 on output
* V600 *acm* 6/05 new flag line_fixname stores whether axis name is changed on input.

* CALLING ARGUMENTS
*	dset_num - the data set fow which a time axis is desired
*	status  - the returning status
	INTEGER*4	dset_num, status
*
* INCLUDE FILES
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'tmap_errors.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xstep_files.cmn_text'
	external xstep_files_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xunits.cmn_text'
	external xunits_data
	include 'xio.cmn_text'
	external xio_data

*
* LOCAL DEFINITIONS
	CHARACTER*80	def_error
	CHARACTER*13	TM_STRING
        CHARACTER*20    buff
	INTEGER*4	scr, nsteps, istp, cpos, cpos1, bpos, bpos0,
     .			i1, i2, i3, i4, i5, i6, iunit, nline,
     .			nsteps_in_file,i			!*kob*
	INTEGER*4	TM_FIND_LIKE_LINE
        INTEGER         TM_GET_CALENDAR_ID, cal_id
	LOGICAL		TM_FPEQ, TM_CHECK_BOX_LOC
	REAL*8		delta, prev_lastep, itt		!*kob* 8/96
	REAL *8		dp_epsilon			!*kob* 5/97
	PARAMETER 	( dp_epsilon = 1.D-10)
	PARAMETER	( scr = 0 )

* Determine the length and delta of the time axis, check regularity, and find
* first timestep
	nsteps = 0
	delta = 0.
        prev_lastep = unspecified_int4
	line_regular(scr) = .TRUE.
	line_start(scr) = unspecified_val8
	DO 100 istp = 1, maxstepfiles
	  IF ( sf_setnum(istp)      .EQ. dset_num .AND.
     .	       sf_aux_set_num(istp) .EQ. 0              ) THEN
* kob add INT and test for sf_lastep to below calculation for linux port
	    IF (sf_lastep(istp) .GT. 0.0D0) THEN
	      nsteps = nsteps +
     .		     INT((sf_lastep(istp)-sf_1step(istp))/sf_delta(istp) 
     .               + dp_epsilon) + 1
	    ELSE 
	      nsteps = nsteps +
     .		     INT((sf_lastep(istp)-sf_1step(istp))/sf_delta(istp) - 
     .               dp_epsilon) + 1
	    ENDIF			
	    IF (delta .EQ. 0) THEN
	      delta = sf_delta(istp)
	    ELSE IF (.NOT. TM_FPEQ(SNGL(delta),SNGL(sf_delta(istp)))) THEN
	      line_regular(scr) = .FALSE.
            ELSE IF ( prev_lastep .NE. unspecified_int4 ) THEN  ! *sh/ko* 6/92
               IF ( .NOT.TM_FPEQ(SNGL(delta),
     .                           SNGL(sf_1step(istp)-prev_lastep)))
     .              line_regular(scr) = .FALSE.
	    ENDIF
	    IF (line_start(scr) .EQ. unspecified_val8) 
     .	        line_start(scr) = sf_1step(istp)
            prev_lastep = sf_lastep(istp)
	  ENDIF
  100	CONTINUE
	line_dim(scr) = nsteps

	IF (line_regular(scr)) THEN
* ... regular
	  line_delta(scr) = delta
	  line_subsc1(scr) = unspecified_int4
	ELSE
* ... irregular
* *kob* 8/96 removed all bpos and bpos0 stuff
	  line_start(scr)   = unspecified_val8
	  line_delta(scr)   = unspecified_val8
	  line_subsc1(scr) = next_line_mem_pos
	  cpos = line_subsc1(scr)
	  cpos1 = cpos
	  IF ((line_subsc1(scr)+2*line_dim(scr)) .GT. maxlinestore)
     .	     GOTO 9000
* Create the time axis and grid box list
* kob - 8/96 - modify so loop not based on itt (now a real) but instead on
*		nsteps_in_file
	  DO 120 istp = 1,maxstepfiles
	    IF ( sf_setnum(istp) .EQ. dset_num .AND.
     .	         sf_aux_set_num(istp) .EQ. 0         ) THEN
	      delta = sf_delta(istp)
*add sanity check for delta *kob* 8/96
	      IF (delta .EQ. 0 ) delta = 1
	      nsteps_in_file = NINT((sf_lastep(istp)-
     .                  sf_1step(istp))/delta) + 1
	      itt = sf_1step(istp)
	      DO 110 i = 1,nsteps_in_file
	        line_mem(cpos) = itt
	        cpos = cpos+1
		itt = itt + delta
  110	      CONTINUE
	    ENDIF
  120	  CONTINUE
	  CALL TM_IRREGAX_MID(line_subsc1(scr),line_dim(scr))
	ENDIF

* Time axis created, fill in the missing pieces
* ... name?
        line_fixname(scr) = .FALSE.
        IF (line_name(scr) .EQ. 'TIME') line_fixname(scr) = .TRUE.
	CALL TM_NEW_LINE_NAME('TIME',line_name(scr))

* ... calendar name?
	line_cal_name(scr) = ds_cal_name(dset_num)

* ...Time_step units
	line_t0(scr)        = ds_t0time(dset_num)
	line_tunit(scr)     = ds_time_unit(dset_num)
	DO 130 iunit = pun_1st_time, pun_last_time
	  IF ( line_tunit(scr) .EQ. un_convert(iunit) ) THEN
	    line_units(scr)     = un_name(iunit)
	    line_unit_code(scr) = iunit
	    GOTO 140
	  ENDIF
  130	CONTINUE
	line_units(scr)     = ' '
	line_unit_code(scr) = 0

* ...Set modulo
 140	line_modulo(scr)    = .FALSE.

* ...Check if time/date string conversion is possible

        cal_id = TM_GET_CALENDAR_ID (line_cal_name(scr))
        
        IF (line_shift_origin(scr)) THEN
           buff = '01-JAN-0001 00:00:00'
        ELSE
 	   buff = line_T0(scr)
        ENDIF

	CALL TM_BREAK_DATE( buff,cal_id,
     .                      i1,i2,i3,i4,i5,i6,status )
	IF ( line_tunit(scr) .LE. 0       .OR.
     .	     line_t0(scr)    .EQ. ' '     .OR.
     .	     status          .NE. merr_ok      ) THEN
	  line_direction(scr) = 'NA'
	  WRITE (tooplu, 1000) ds_name(dset_num)
 1000	  FORMAT 
     .  (' Warning: missing or improper date/time format for T0 in ',A)
	ELSE
	  line_direction(scr) = 'TI'
	ENDIF

* ...Compare this with existing axis
	CALL TM_GET_LIKE_LINE(scr,line_name(scr),nline,status)
	IF (status .NE. merr_ok) GOTO 9200
* ...Warn about possible gaps in axis
* Bug fix - change next statment from compound if to if then ---kob 10/8/91
	IF (.NOT. line_regular(nline)) THEN
           IF (.NOT. TM_CHECK_BOX_LOC(nline) ) GOTO 9100
        ENDIF
!	IF (.NOT. line_regular(nline) .AND.  
!  .	 .NOT. TM_CHECK_BOX_LOC(nline) ) GOTO 9100

* Time axis complete
	GT_GET_TIME_AXIS = nline

* If necessary - find next free line memory storage position
	IF (.NOT. line_regular(nline))
     .	 next_line_mem_pos = next_line_mem_pos + 2*line_dim(nline) + 1
	status = merr_ok
	GOTO 9999

 9000	CALL TM_ERRMSG (merr_linstorlim, status,'GT_GET_TIME_AXIS',
     .			no_descfile, no_stepfile,
     .			'MAX='//TM_STRING(DBLE(maxlinestore)),
     .			no_errstring, *9999)

 9100	CALL TM_ERRMSG (merr_badlinedef, status, 'GT_GET_TIME_AXIS',
     .			no_descfile, no_stepfile,
     .			'A gap or overlap exists in time axis',
     .			no_errstring, *9999)

 9200	CALL TM_ERRMSG (merr_linelim, status, 'GT_GET_TIME_AXIS',
     .			no_descfile, no_stepfile,
     .			'MAX='//TM_STRING(DBLE(max_lines)), no_errstring,
     .			*9999)

* Exit subroutine
 9999	RETURN
	END
