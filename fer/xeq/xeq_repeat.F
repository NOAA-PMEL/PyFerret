	SUBROUTINE XEQ_REPEAT

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
* sets up the control stack (COMMON/~CONTROL/) to repeat the indicated
* command for a range of default contexts
* eg. "REPEAT/T=27740:35000:10 CONTOUR/@N/K=1 TEMP"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/29/87
* revision 0.1 - 9/16/87 - unknown_qual_bad moved to FERRET.PARM
* revision 0.2 - 9/25/87 - added "d" notation for REPEAT limits
* revision 0.3 - 4/11/88 - special checks on T axis for calendar notation
* revision 0.4 - 7/27/88 - added default context to GET_CONTEXT_MODS call
* revision 0.5 -10/31/88 - limits reversed for negative delta
* V200:  6/15/89 - eliminated "d" notation and "relative" logic
* Unix/RISC port - 1/91 - cant use "/star" in documentation
* V230:  7/23/92 - added cs_save_* to restore context after REPEAT loops
*       - add check on cx_calendar rather than cx_lo/hi_ww to indicate 
*         whether time is date or timesteps
* 12/97 *kob* - use new logical cs_is_calendar
* V554 1/04 *acm* - additions for /RANGE repeat loop control

	include	'ferret.parm'
	include	'command.parm'
	include 'errmsg.parm'
	include 'slash.parm'
	include 'xcontext.cmn'
	include 'xcontrol.cmn'
	include 'xprog_state.cmn'
	include 'xtext_info.cmn'
	include 'xplot_state.cmn'
	include 'xplot_setup.cmn'
	include	'gkscm1_inc.decl'	! with gks common
	include	'GKSCM1.INC'            ! with gks common
        include 'rpn.parm'              ! with maxatoms
        include 'xalgebra.cmn'

* local variable declarations
	LOGICAL	by_date, report_unknown, range
	INTEGER	INDEX, i, idim, limits_cx, status, csp1, rlo, rhi, rdel
	REAL*8	temp
 
	INTEGER STR_UPCASE, loc, equal_pos, last_pos, dflt_cx, cvar,
     .          istart, iflag, locr, cat, var
	REAL val
	CHARACTER GKS_X_CONID*16, anim_win_id*16, counter_name*128
	LOGICAL loop

* check that something was given to repeat
	IF ( num_args .EQ. 0 ) CALL ERRMSG
     .			( ferr_invalid_command, status, 'REPEAT what ?', *5000 )


* Did they specify /NAME without /RANGE=?

          loc = qual_given(slash_repeat_name) 
          IF (loc .GT. 0) THEN
             locr = qual_given(slash_repeat_range) 

	     IF (locr .LE. 0) GO TO  5100

          ENDIF

*acm 1/01 check to see if we are using /RANGE=
       loc = qual_given(slash_repeat_range) 
       range = (qual_given(slash_repeat_range) .GT. 0)
       IF (range) THEN
          idim = 1

           last_pos = qual_end(loc)
           equal_pos = 
     .          INDEX(cmnd_buff(qual_start(loc):last_pos), '=') +
     .          qual_start(loc) - 1


* Did they specify /RANGE and also /I/J/K/L/X/Y/Z/T ?
	   report_unknown = .FALSE.
	   limits_cx = cx_stack_ptr_base + 1
	   CALL INIT_CONTEXT_MODS ( limits_cx )
	   CALL GET_CONTEXT_MODS ( cx_last,
     .				cmnd_buff,
     .				limits_cx,
     .				max_qual_list,
     .				num_qualifiers,
     .				qual_start,
     .				qual_end,
     .				report_unknown,
     .				status	)
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   DO 100 i = 1,4
	      IF ( cx_given(i,limits_cx) ) GOTO 5300
 100	   CONTINUE

* Are we using a name for the repeat counter?
          counter_name = 'REPCOUNT'

          loc = qual_given(slash_repeat_name) 
          IF (loc .GT. 0) THEN

	     CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        counter_name, status )

             IF ( status .NE. ferr_ok ) GOTO 5000

* Check for legal name: not already defined, not a pseudo-var

	     CALL FIND_VAR_NAME ( pdset_irrelevant, counter_name, 
     .                            cat, var )
             IF ( cat .EQ. cat_pseudo_var   ) GOTO 5200
             IF ( cat .EQ. cat_counter_var   ) GOTO 5400

          ENDIF

* allocate next spot for the given limits on the control stack
* ... but leave the stack in place in case of errors in this routine

	   CALL STACK_PTR_UP( csp, cs_max, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   csp1 = csp
	   CALL STACK_PTR_DN( csp, 0, status )

* File the expression and name in the counter-variable area and set up the
* context so the command stack can be loaded 

           istart = 1
	   CALL INIT_COUNTER_VAR ( counter_name, cmnd_buff(equal_pos+1:last_pos), 
     .             istart, last_pos-equal_pos, rlo, rhi, rdel, cvar, status )\

	   IF ( status .NE. ferr_ok ) RETURN

* load the start and end limits and the increment

           cs_by_ss( csp1 ) = .TRUE.
	   cs_delta( csp1 ) = rdel
	   cs_limit( csp1 ) = rhi
	   cs_phase( csp1 ) = rlo
	
* error message if delta is 0

           IF (cs_delta( csp1 ) .EQ. 0) CALL ERRMSG( ferr_invalid_command, status,
     .			'Delta value on REPEAT must be nonzero', *5000 )

* save the current context so it can be restored afterwards

           cs_save_by_ss( csp1 ) = cx_by_ss(idim,cx_last)
           IF ( cs_save_by_ss(csp1) ) THEN
              cs_save_lo( csp1 ) = cx_lo_ss (cx_last,idim)
              cs_save_hi( csp1 ) = cx_hi_ss (cx_last,idim)
           ELSE
              cs_save_lo( csp1 ) = cx_lo_ww (idim,cx_last)
              cs_save_hi( csp1 ) = cx_hi_ww (idim,cx_last)
           ENDIF

* Save name of REPEAT/RANGE variable

           csp_cvar(cvar) = csp1
           IF (counter_name .EQ. 'repcount') THEN
              cs_repeatname( csp1 ) = 'REPCOUNT'
           ELSE
              iflag = STR_UPCASE( cs_repeatname( csp1 ), counter_name )
           ENDIF

* stack bump  ???
	   CALL STACK_PTR_UP( csp, cs_max, status )

* load the axis to which the limits apply
	   cs_axis( csp ) = idim

* load the command line to be repeatedly executed
	   cs_text( csp ) = cmnd_buff( arg_start(1):len_cmnd )

* and tell 'em who owns this level
	   cs_cmnd_num( csp ) = cmnd_repeat

* and set the default context in preparation for first call
!	   CALL CS_SET_CONTEXT  !??? Not for REPEAT/RANGE

* success - tell 'em we've taken over and start workin'
	   cs_in_control = .TRUE.
	   input_source  = input_from_cs

* kob 1/01 - check to see if there is a loop in the repeat command.  if so, set
*            cs_loop accordingly
	   loc = qual_given(slash_repeat_loop)
	   IF (loc .GT. 0) THEN
	      loop = .TRUE.
	      cs_start(csp1) = cs_phase(csp1)
	      CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .	           val, status ) ! syntax check
	      IF ( val .NE. unspecified_val4 .AND. val .NE. 0) THEN
                 cs_loop(csp1) = INT(val)
	      ELSE
                  cs_loop(csp1) = 1
	      ENDIF
	   ENDIF
*kob 1/01 check to see if we are in an animation repeat loop
	   animate = qual_given(slash_repeat_animate) .GT. 0
           IF (animate) THEN
* set first_frame to true - we don't need to animate first frame - its
*   already in the plot window
	      first_frame = .TRUE. 
	   ENDIF

        ELSE  ! not /RANGE

* get the limits for repeating
*  eg. "REPEAT/X=160E:140W:10 LIST/X=-5D:+5D/Y=0D:5D TEMP"
	   limits_cx = cx_stack_ptr_base + 1
* *kob* 1/00 - dont report unknown command qualifiers
	   report_unknown = .FALSE.
	   CALL INIT_CONTEXT_MODS ( limits_cx )


	   CALL GET_CONTEXT_MODS ( cx_last,
     .				cmnd_buff,
     .				limits_cx,
     .				max_qual_list,
     .				num_qualifiers,
     .				qual_start,
     .				qual_end,
     .				report_unknown,
     .				status	)

	   IF ( status .NE. ferr_ok ) GOTO 5000

* determine the (unique) axis of REPEATing
	   idim = unspecified_int4
	   DO 200 i = 1,4
	      IF ( cx_given(i,limits_cx) ) THEN
	         IF ( idim .EQ. unspecified_int4 ) THEN
	            idim = i
	         ELSE
	            CALL ERRMSG( ferr_invalid_command, status,
     .			'Use nested REPEAT for multiple axes', *5000 )
	         ENDIF
 	      ENDIF
 200	   CONTINUE
	   IF ( idim .EQ. unspecified_int4 ) CALL ERRMSG
     .		(ferr_invalid_command,status,'REPEAT on which axis ?',*5000)

* allocate next spot for the given limits on the control stack
* ... but leave the stack in place in case of errors in this routine
	   CALL STACK_PTR_UP( csp, cs_max, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   csp1 = csp
	   CALL STACK_PTR_DN( csp, 0, status )

* load the start and end limits and the increment

	   cs_by_ss( csp1 ) = cx_by_ss( idim, limits_cx )
           IF ( cx_delta( idim, limits_cx ) .EQ. unspecified_val8 ) THEN
	      cs_delta( csp1 ) = 1
	   ELSE
	      cs_delta( csp1 ) = cx_delta( idim, limits_cx )
	   ENDIF
	   IF ( cs_by_ss( csp1 ) ) THEN
	      cs_limit( csp1 ) = cx_hi_ss( limits_cx, idim )
	      cs_phase( csp1 ) = cx_lo_ss( limits_cx, idim )
	   ELSE
	      cs_limit( csp1 ) = cx_hi_ww( idim, limits_cx )
	      cs_phase( csp1 ) = cx_lo_ww( idim, limits_cx )
           ENDIF

* error message if delta is 0

           IF (cs_delta( csp1 ) .EQ. 0) CALL ERRMSG( ferr_invalid_command, status,
     .			'Delta value on REPEAT must be nonzero', *5000 )

           cs_repeatname( csp1 ) = unspecified_name4

* save the current context so it can be restored afterwards

           cs_save_by_ss( csp1 ) = cx_by_ss(idim,cx_last)
           IF ( cs_save_by_ss(csp1) ) THEN
              cs_save_lo( csp1 ) = cx_lo_ss (cx_last,idim)
              cs_save_hi( csp1 ) = cx_hi_ss (cx_last,idim)
           ELSE
              cs_save_lo( csp1 ) = cx_lo_ww (idim,cx_last)
              cs_save_hi( csp1 ) = cx_hi_ww (idim,cx_last)
           ENDIF

* special check for the validity of time axis limits: calendar dates are
* stored as negative seconds since 1-jan-0000:00:00:00
* 8/97 *kob* use cx_calender to check if time is date rather than time step
* 12/97 *kob* set new logical cs_is_calendar now as well
	   by_date = ( idim .EQ. t_dim )
     .	   .AND. ( .NOT.cx_by_ss(idim,limits_cx) )
     .	   .AND. ( cx_calendar(limits_cx) )
* ... delta value in hours: hours-->seconds (to add to secs from BC)
	   IF ( by_date ) THEN
               cs_delta(csp1) =  -3600. * cs_delta(csp1)
	       cs_is_calendar(csp1) = .TRUE.
	   ENDIF

* reverse limits if negative delta is desired
	   IF (( ( .NOT.by_date .AND. (cs_delta(csp1) .LT. 0.0D0) )
     .   	.OR. (      by_date .AND. (cs_delta(csp1) .GT. 0.0D0) ) ) ) THEN
	      temp = cs_phase(csp1)
	      cs_phase(csp1) = cs_limit(csp1)
	      cs_limit(csp1) = temp
	   ENDIF

* from here on no errors can occur so make the stack bump permanent
	   CALL STACK_PTR_UP( csp, cs_max, status )

* load the axis to which the limits apply
	   cs_axis( csp ) = idim

* load the command line to be repeatedly executed
	   cs_text( csp ) = cmnd_buff( arg_start(1):len_cmnd )

* and tell 'em who owns this level
	   cs_cmnd_num( csp ) = cmnd_repeat

* and set the default context in preparation for first call
	   CALL CS_SET_CONTEXT

* success - tell 'em we've taken over and start workin'
   	   cs_in_control = .TRUE.
	   input_source  = input_from_cs

* kob 1/01 - check to see if there is a loop in the repeat command.  if so, set
*            cs_loop accordingly
	      loc = qual_given(slash_repeat_loop)
	      IF (loc .GT. 0) THEN
	         loop = .TRUE.
	         cs_start(csp1) = cs_phase(csp1)
	         CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .	              val, status ) ! syntax check
	         IF ( val .NE. unspecified_val4 .AND. val .NE. 0) THEN
	   	    cs_loop(csp1) = INT(val)
	         ELSE
		    cs_loop(csp1) = 1
	         ENDIF
	      ENDIF
*kob 1/01 check to see if we are in an animation repeat loop
	   animate = qual_given(slash_repeat_animate) .GT. 0
           IF (animate) THEN
* set first_frame to true - we don't need to animate first frame - its
*   already in the plot window
	     first_frame = .TRUE. 
	   ENDIF

        ENDIF  ! IF (range)

	RETURN

* error exit
* ... control stack pointer, CSP, corrected in GFD_ERROR_HANDLER
 5000	RETURN

 5100	CALL ERRMSG( ferr_invalid_command, status,
     .		'REPEAT/NAME requires /RANGE', *5000 )
 5200   CALL ERRMSG( ferr_syntax,status,
     .  'cannot use pseudo-variable name for repeat counter: '
     .              //counter_name,*5000 )
 5300   CALL ERRMSG( ferr_syntax,status,
     .  'REPEAT/RANGE cannot be used with an axis qualifier '//cmnd_buff ,*5000 )

 5400   CALL ERRMSG( ferr_syntax,status,
     .  'nested REPEAT loops with the same counter name '
     .             //counter_name,*5000 )

	END


