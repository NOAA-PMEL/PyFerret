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
*                  [IN FACT THIS TURNS OUT TO BE IRRELEVANT ... ]
* v522 *kob* 12/00 - add xplot_setup common file to check to see if we
*                    are animating
* v540 *kob* 1/02 - fix bug in which last animation plot wasn't being
*                   flushed to the screen
* v541 *kob* 8/02 - for animating, check for Xserver backingstore and
*                   don't animate if it's not set
* V554 1/04 *acm* - add cs_repeatname for named /RANGE loop control
* V600 5/06 *acm* - the call to new_gks_window needs window_title.
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)

        IMPLICIT NONE
* calling argument declarations:
        CHARACTER*(*)   cmnd_line
* RETURN 1 - used when no command can be supplied

* internal variable declarations:
        INTEGER prev, status, limit, i, nparen
        CHARACTER CX_DIM_STR*40
        INTEGER  TM_LENSTR1, slen
        CHARACTER*10 TM_FMT, out_buff
        REAL    val
        LOGICAL range

	include 'tmap_dims.parm'
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
        include 'GKSCM1.INC'            ! with gks common
	include 'xtoday.cmn'		! with window_title

        IF ( interrupted ) THEN

* *kob* if we are animating, stop
           IF ( animate ) THEN
              animate = .FALSE.
              first_frame = .TRUE.
           ENDIF
           CALL ERRMSG( ferr_interrupt,status,' ', *1000 )

        ELSEIF ( cs_cmnd_num( csp ) .EQ. cmnd_repeat ) THEN

***** REPEAT loop *******
* done yet ?
           range = .FALSE.
           IF ( (cs_limit(csp)-cs_phase(csp))*cs_delta(csp)
     .          .LT. 0.0 ) THEN
* ... yes
              IF ( animate .AND. (wsid .GT. 0) ) THEN
*                make sure the latest image is displayed
                 IF ( wn_open(wsid) ) THEN
                    CALL FGD_GUWK(wsid, 0)
                 ENDIF
              ENDIF
* *kob* check to see if we need to keep looping through animation
              IF ( cs_loop(csp) .GT.1 ) THEN 
                 cs_phase( csp ) = cs_start( csp ) 
                 cs_loop( csp ) = cs_loop( csp) - 1 
                 cmnd_line = cs_text( csp ) 
                 CALL CS_SET_CONTEXT 
                 RETURN 1
              ELSE
* *kob* repeat loop done so we are done animating.
                 IF ( animate ) THEN
                    animate = .FALSE.
                    first_frame = .TRUE.
                 ENDIF
                 CALL POP_CS
                 RETURN 1
              ENDIF
           ELSE
*       ... no - get the next command
              IF ( animate ) THEN
*                make sure the latest image is displayed
                 IF ( wsid .GT. 0 ) THEN
                    IF ( wn_open(wsid) ) THEN
                       CALL FGD_GUWK(wsid, 0)
                    ENDIF
                 ENDIF
*                check if we are past commands for the first frame
                 IF ( first_frame ) THEN
                    IF ( ( (cs_phase(csp) - cs_start(csp))
     .                         * cs_delta(csp) .GT. 0.0 ) ) THEN
                       first_frame = .FALSE.
                    ENDIF
                 ENDIF
              ENDIF
              cmnd_line = cs_text( csp )
* ... set up the default context to evaluate the command
              range = (cs_repeatname( csp ) .NE. unspecified_name4)
              IF (.NOT. range) THEN
                 CALL CS_SET_CONTEXT
              ELSE
                 val = cs_phase( csp )
                 out_buff = TM_FMT(val,10,10,len_rbuff)
              ENDIF

* ... prepare for next time around
              cs_phase( csp ) = cs_phase( csp ) + cs_delta( csp )
           ENDIF

* ... echo the progress so the operator can monitor
           IF ( mode_verify ) THEN
              IF (.NOT. range) THEN
                 risc_buff = CX_DIM_STR( cs_axis(csp), cx_last, ':',
     .                                   .TRUE., len_rbuff )
              ELSE
                 slen = TM_LENSTR1(cs_repeatname(csp))
                 risc_buff = cs_repeatname(csp)(1:slen)//':'//out_buff(1:len_rbuff)
                 len_rbuff = TM_LENSTR1(risc_buff)
              ENDIF

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
              cs_phase(csp) = 1         ! signal to POP after this command

           ENDIF

* ***** unrecognized command in control of stack

        ELSE
           CALL ERRMSG( ferr_internal, status, 'command stack', *1000 )
        ENDIF

 1000   RETURN
        END
