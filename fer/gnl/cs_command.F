	SUBROUTINE CS_COMMAND( cmnd_line, * )

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
* this routine is called to get the next command when the program is under
* "internal" control (one command executing another)
* its function is to return the next command to execute or to flag that the
* internal level controlling the program has completed

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.0 - 4/29/87
* revision 0.1 - 4/12/88 - allowed negative delta for time axis
* v300: 1/2/93 - incorporated cmnd_semicolon
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* V420:  4/25/96 - added IF command
*
* 1/98 sh&kob
* "KNOWN BUG: The proper formatting of the REPEAT limit requires a 
*              knowledge of the grid of the result(see lohi_string.F). 
*              Since the grid is not determined until the first repeat 
*              loop is executed the formatting of the first loop may be
*              incorrect. Fixing this would require that a new set of 
*              world coordinate formatting routines be developed that 
*              do not depend on the grid. This seemed too much trouble 
*              given the small consequeunces."

* V510 *sh* 3/00 - interrupts need to be processed in here!!
*		   [IN FACT THIS TURNS OUT TO BE IRRELEVANT ... ]
* v522 *kob* 12/00 - add xplot_setup common file to check to see if we
*                    are animating
* v540 *kob* 1/02 - fix bug in which last animation plot wasn't being
*                   flushed to the screen
* v541 *kob* 8/02 - for animating, check for Xserver backingstore and
*                   don't animate if it's not set
* calling argument declarations:
	CHARACTER*(*)	cmnd_line
* RETURN 1 - used when no command can be supplied

* internal variable declarations:
	INTEGER	prev, status, limit, i, nparen
        CHARACTER CX_DIM_STR*40,anim_win_id*16,GKS_X_CONID*16
	INTEGER  anim_id, bs, ws_id
	
#ifdef unix
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'command.parm'
	include 'xprog_state.cmn'
	include 'xcontext.cmn'
	include 'xcontrol.cmn'
        include 'xrisc.cmn'
	include 'xinterrupt.cmn'
	include 'xplot_setup.cmn'
	include 'xplot_state.cmn'
	include	'gkscm1_inc.decl'	! with gks common
	include	'GKSCM1.INC'	                ! with gks common

#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:COMMAND.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XCONTROL.CMN'
	INCLUDE 'FERRET_CMN:XRISC.CMN'
#endif

	IF ( interrupted ) THEN
* *kob* if we are animating, stop and reset active window
	   IF (animate .EQV. .TRUE.) THEN
	      animate = .FALSE.
	      CALL SWAP_ACTIVE_WINDOW
	   ENDIF
	   CALL ERRMSG( ferr_interrupt,status,' ', *1000 )

	ELSEIF ( cs_cmnd_num( csp ) .EQ. cmnd_repeat ) THEN

***** REPEAT loop *******
* done yet ?
	   IF ( (cs_limit(csp)-cs_phase(csp))*cs_delta(csp)
     .          .LT. 0.0 ) THEN
* ... yes
* *kob* check to see if we need to keep looping through animation
	      IF ( cs_loop(csp) .GT.1 ) THEN 
		 cs_phase( csp ) = cs_start( csp ) 
		 cs_loop( csp ) = cs_loop( csp) - 1 
		 cmnd_line = cs_text( csp ) 
		 CALL CS_SET_CONTEXT 
		 RETURN 1
	      ELSE
* *kob* repeat loop done so we are done animating.  reset active window
* 1/02 v5.4 - and also flush last plot to screen
		 IF (animate .EQV. .TRUE.) THEN
		    animate = .FALSE.
* flush last plot to screen
		    wsid = win_id9
		    anim_id = active_win_id
		    CALL COPY_BUFFERED_WINDOW(wsid, anim_id)
		    CALL CLEAR_WINDOW(wsid)
* reset active window
		    CALL SWAP_ACTIVE_WINDOW
		 ENDIF
		 CALL POP_CS
		 RETURN 1
	      ENDIF
	   ELSE
*       kob 12/00 - before we grab the next command, if we are animating lets
*             take the snapshot here
	      IF (animate .EQV. .TRUE.) THEN
		 IF (.NOT. first_frame) THEN
*       kob 8/02 - need to check and see if the X server has
*                  backing store turned on.  If not, give an
*                  error and stop animating.
		    DO ws_id = 1,max_windows
		       IF (wn_open(ws_id) ) GOTO 50
		    end do
		    CALL ERRMSG( ferr_invalid_command, status, 
     .                              'No graphics window open', *1000 )
 50		    CALL BACKING_STORE_ENABLED(ws_id, bs)
		    IF (bs .EQ. 1) THEN
		       IF (.NOT. wn_open(win_id9)) THEN 
*       * open an unmapped window
			  CALL UNMAP_ANIMATE_WINDOW
			  anim_win_id = GKS_X_CONID( win_id9 )
			  CALL NEW_GKS_WINDOW( win_id9, anim_win_id, status ) 
			  if (status .EQ. ferr_ok) wn_open(win_id9) = .TRUE.
			  CALL SETUP_LINE_BUNDLES( win_id9, wstype )
*       swap active windows so the unampped window becomes the active
*       window, and the newly created window is the animating window
			  CALL SWAP_ACTIVE_WINDOW
		       ELSE 
			  wsid = win_id9
			  anim_id = active_win_id
			  CALL COPY_BUFFERED_WINDOW(wsid, anim_id)
*       need to clear the double buffered unmapped window before next plot *kob*
			  CALL CLEAR_WINDOW(wsid)
		       ENDIF
		    ELSE
*       kob 8/02 - X server backing store not turned on so we
*                  can't animate
		       animate = .FALSE.
		       CALL ERRMSG( ferr_not_implemented, status, 
     .                  'Animations require X-server backingstore', *1000 )

		    ENDIF   
		 ELSE
		    first_frame = .false.
		 ENDIF   
	      ENDIF
*       ... no - get the next command
	      cmnd_line = cs_text( csp )
* ... set up the default context to evaluate the command
	      CALL CS_SET_CONTEXT
* ... prepare for next time around	   
	   cs_phase( csp ) = cs_phase( csp ) + cs_delta( csp )
	   ENDIF

* ... echo the progress so the operator can monitor
           IF ( mode_verify ) THEN
              risc_buff = CX_DIM_STR( cs_axis(csp), cx_last, ':',
     .             .TRUE., len_rbuff )
	      CALL SPLIT_LIST(pttmode_ops, ttout_lun, '!-> REPEAT: '
     .                           //risc_buff(1:len_rbuff), 12+len_rbuff)
           ENDIF

	ELSEIF ( cs_cmnd_num(csp) .EQ. cmnd_semicolon ) THEN

***** semicolon-separated command group
* initialize the extracting of a single command
           prev  = cs_phase(csp)
           limit = cs_limit(csp)

* done yet?
           IF ( prev .GE. limit ) THEN
* ... yes
	      CALL POP_CS
	      RETURN 1

	   ELSE
* ... no - get the next command (find semicolon NOT enclosed in parens)
              nparen = 0
              DO 100 i = prev+1, limit
                 IF ( cs_text(csp)(i:i) .EQ. '(' ) THEN
                    nparen = nparen + 1
                 ELSEIF ( cs_text(csp)(i:i) .EQ. ')' ) THEN
                    nparen = nparen - 1
                 ELSEIF ( cs_text(csp)(i:i) .EQ. ';' ) THEN
* ... found another semicolon-terminated command in group
                    IF ( nparen .EQ. 0 ) THEN
                       cmnd_line = cs_text(csp)(prev+1:i-1)
                       cs_phase(csp) = i
                       GOTO 1000
                    ENDIF
                 ENDIF
 100          CONTINUE
* ... last command in buffer
              cmnd_line = cs_text(csp)(prev+1:limit)
              cs_phase(csp) = limit
                   
	   ENDIF

* ****** conditional "IF" command

	ELSEIF ( cs_cmnd_num(csp) .EQ. cmnd_if ) THEN

* already done?
	   IF (cs_phase(csp) .GT. 0 ) THEN

* ... yes get outta here
	      CALL POP_CS
	      RETURN 1
	   ELSE

* ... no - get the next command
	      cmnd_line = cs_text( csp )
	      cs_phase(csp) = 1		! signal to POP after this command

	   ENDIF

* ***** unrecognized command in control of stack

        ELSE
           CALL ERRMSG( ferr_internal, status, 'command stack', *1000 )
        ENDIF

 1000   RETURN
	END
