	SUBROUTINE XEQ_PPLUS

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
* access program PPLUS ( the underlying graphics package ) directly - either
* by sending a single command to it or by entering the PPLUS command interpreter

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 1/7/87
* revision 0.1 - 9/8/88 - added /RESET option
* revision 0.2 - 1/31/89 - added PPLUS interruptibility
* Unix/RISC port - 10/9/91 - check to see if user changes ylen or yorg

#ifdef unix
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xplot_setup.cmn'    ! added 10/9/91
        include 'axis_inc.decl'  ! added 10/9/91 for ylen,yorg
        include 'AXIS.INC'            ! added 10/9/91 for ylen,yorg
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
	INCLUDE 'XPLOT_SETUP.CMN'    ! added 10/9/91
        INCLUDE 'PPLV11INC:AXIS.INC'            ! added 10/9/91 for ylen,yorg
        INCLUDE 'PPLV11INC:AXIS.INC'            ! added 10/9/91 for ylen,yorg
#endif
	
* local variable declarations:
        REAL temp_ylen, temp_yorg

* local parameter declarations:
	INTEGER		slash_reset
	PARAMETER     ( slash_reset = 0+1 )

* turn on PLOT+ ?
	CALL START_PPLUS

* save ylen and yorg in case the user manually changes those values in PLOT+
* 10/9/91 addition 
        temp_ylen = ylen
        temp_yorg = yorg

	IF ( qual_given( slash_reset ) .GT. 0 ) THEN
* ... reset plot environment to defaults
	   CALL DISP_RESET
	ELSEIF ( num_args .GT. 0 ) THEN
* ... pass a single command to PLOT5
	   CALL PPLCMD (' ',' ',0,cmnd_buff(arg_start(1):len_cmnd),1,1)
	ELSE
* ... enter PPLUS interactively
#ifdef unix
	   CALL PPLCMD ( '/dev/tty', ' ', 0, ' ', 1, 0 )
#else
	   CALL PPLCMD ( 'TT:', ' ', 0, ' ', 1, 0 )
#endif
	   ppl_interrupted = .FALSE.			! since interactive
	ENDIF

* check to see if ylen or yorg has been changed
* if so, parameters saved prior to distorting yaxis are invalidated
        IF ( ylen .NE. temp_ylen ) ppl_saved_yaxlen = ylen
        IF ( yorg .NE. temp_yorg ) ppl_saved_yorg   = yorg

	RETURN

	END

