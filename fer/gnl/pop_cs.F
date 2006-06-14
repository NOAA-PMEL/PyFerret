	SUBROUTINE POP_CS

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
* POP the contral stack turning control over to the next lower level, if any

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.0 - 4/29/87
* V230: restore original context following REPEAT command
* V300 - 2/2/93 - added semicolon-separated command groups
* V420 - added IF (single line version controls input source)
*      - generate warning if multi-line IF is unclosed
* V500 4/99 *sh* - remove the informative message about an unclosed IF
*    This is not an error in IF cond THEN stmt; QUIT ENDIF 
* V530:  8/00 *sh* - when restoring the original context after a REPEAT it
*	is necessary to set the unsaved region (as per cx_by_ss) to unspecified
*	(The need to do this represents a bug somewhere up the line -- region
*		information is used without regard to cx_by_ss)
*	see err522_repeat_leftover.jnl
* V554 1/04 *acm* - add cs_repeatname for named /RANGE loop control; reset repeat controls.
* v560 3/04 *acm    reset last_cmnd_buff  when resetting repeat controls.
*      6/04 ywei    change MATCH_NAME function
* v5.80 *acm* 11/04 use cx_cal_id to store the calender used when
*                   computing world coordinates; restore cx_cal_id
*                   along with  the rest of the context after a REPEAT
* V600  4/06 *acm* - additional declarations found by Solaris compiler

* internal variable declarations:
	INTEGER	LIB_FREE_LUN,stat,axis,ivar,cvar
        INTEGER j, num_indices

	include 'ferret.parm'
	include 'command.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'
	include 'xcontrol.cmn'
        include 'xcontext.cmn'
        include 'xalgebra.cmn'
        include 'xvariables.cmn'


* internal error ?
	IF ( csp .LT. 1 ) STOP 'csp_undflw'

* popping out of a stack level with an unclosed IF statement is a form of error
* we will issue a warning and then implicitly close the IF
	IF (if_conditional) THEN
	   IF (csp .LE. if_cs(ifstk) ) THEN
! no! 4/99	      CALL WARN('Unclosed IF statement ... assuming ENDIF')
	      ifstk = ifstk - 1
	      IF (ifstk .EQ. 0 ) if_conditional = .FALSE.
	   ENDIF
	ENDIF

* clean up old level
	IF ( cs_cmnd_num( csp ) .EQ. cmnd_go ) THEN
* ... close input GO file
* back out of yonghua's routine call because of 64bit problems *kob**     
*      CALL my_close(cmnd_lun)
	   CLOSE( UNIT = cmnd_lun, ERR = 10 )
 10	   stat = LIB_FREE_LUN( cmnd_lun )
	ELSEIF ( cs_cmnd_num( csp ) .EQ. cmnd_repeat ) THEN
* ... restore context prior to REPEAT loop (along repeat axis only)
           axis = cs_axis( csp )
           cx_by_ss(axis,cx_last) = cs_save_by_ss( csp )

           IF (axis .eq. t_dim) THEN
              cx_calendar(cx_last) = .FALSE.
              cx_cal_id(cx_last) = unspecified_int4
           ENDIF

           IF ( cs_save_by_ss( csp ) ) THEN
              cx_lo_ss (cx_last,axis) = cs_save_lo( csp )
              cx_hi_ss (cx_last,axis) = cs_save_hi( csp )
              cx_lo_ww (axis,cx_last) = unspecified_val8
              cx_hi_ww (axis,cx_last) = unspecified_val8
           ELSE
              cx_lo_ww (axis,cx_last) = cs_save_lo( csp )
              cx_hi_ww (axis,cx_last) = cs_save_hi( csp )
              cx_lo_ss (cx_last,axis) = unspecified_int4
              cx_hi_ss (cx_last,axis) = unspecified_int4
              cx_cal_id (cx_last) = cs_save_cal(csp) 
           ENDIF

* clean up counter variable 

           cvar = 0
*           len_test = TM_LENSTR(cs_repeatname(csp))
*	   DO 20 ivar =  num_counter_vars, 1, -1
*              len_mchars = TM_LENSTR(countervar_name(ivar))
*	      IF ( MATCH_NAME( cs_repeatname(csp), len_test,
*     .           countervar_name(ivar), len_mchars ) ) THEN
*                cvar = ivar
*                GOTO 1000
*              ENDIF
*  20	   CONTINUE
*1000       CONTINUE
 	CALL string_array_find(countervar_name_head,
     .                         cs_repeatname(csp),
     .                         LEN(cs_repeatname(csp)),
     .                         string_array_result(1),
     .                         max_uvar,
     .                         num_indices)
        IF(num_indices.GT.0) THEN
              cvar = string_array_result(1)
        ENDIF

           IF (cvar .GT. 0) THEN
*              countervar_name(cvar) = unspecified_name4
              CALL string_array_modify(countervar_name_head,
     .                                 cvar,
     .                                 unspecified_name4,
     .                                 LEN(unspecified_name4))

              alg_cvar( cvar ) = unspecified_name4
              cs_repeatname( csp ) = unspecified_name4
              csp_cvar( csp ) = unspecified_int4
              last_cmnd_buff = ' '
           ENDIF

	ENDIF
	csp = csp - 1

* now who gets control ?
	IF ( csp .EQ. 0 ) THEN
* ... the operator
	   cs_in_control = .FALSE.
	   input_source  = input_from_tty
	ELSE
* ... the next stack level down
	   IF    ( cs_cmnd_num( csp ) .EQ. cmnd_go     ) THEN
	      input_source = input_from_file
	      cmnd_lun = cs_lun( csp )
	   ELSEIF ( cs_cmnd_num( csp ) .EQ. cmnd_semicolon
     .         .OR. cs_cmnd_num( csp ) .EQ. cmnd_if
     .         .OR. cs_cmnd_num( csp ) .EQ. cmnd_repeat ) THEN
	      input_source = input_from_cs
	   ELSE
	      STOP 'cs_stk_??'
	   ENDIF
	ENDIF
		   
	RETURN
	END




