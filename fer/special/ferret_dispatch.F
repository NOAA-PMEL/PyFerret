	SUBROUTINE FERRET_DISPATCH( memory, command, rtn_flags, nflags,
     .					rtn_chars, nchars, nerrlines )

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
* Execute a FERRET command
* WHILE MODE GUI is "CANCELLED" control will remain inside this routine
* If "command" contains text that command will be executed immediately
* (formatted as a c-style, null-terminated string)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program


* FERRET program history:
* initially tailored to output format and content of the Philander/Seigel 
* model from GFDL
* revision 0.0  - 4/3/86
* revision 1.0 - 11/17/86 - first "official" release
* revision 2.0 - 10/23/87 - "official" release
* revision 2.01(temporary) - 10/23/87 - smaller memory size, bug fixes,
*			2 typos in XVARIABLES_DATA, ^C added, ZT planes added
* revision 2.02 - ?????
* revision 2.10 - 5/6/88 - "final" release - /NODEBUG version
* FERRET 1.00     - 6/10/88 - rename of GFDL 2.10
* FERRET 1.10     -  8/2/88 - numerous bug fixes and enhancements
* FERRET 1.20     - 2/17/89 - numerous bug fixes and enhancements
* FERRET 1.21     - 4/19/89 - minor bug fixes
* FERRET 2.00	  - 5/??/89 - internal re-write: 4D grids and "object oriented"
*			      transformations
* FERRET 3.00     - 1/29/93 - revision 2.2-->2.3 changes became so extensive
*                             and prolonged it made sense to rename to V3.0
* FERRET 3.10     - 4/94 - official release using XGKS
* FERRET 3.11     - 5/94 - added FILE/ORDER=/FORMAT=STREAM
* FERRET 3.12     - 5/94 - restructured to be "dynamic memory" (C main routine)
*			   former MAIN became FERRET_DISPATCH routine

* FERRET_DISPATCH routine history
* as MAIN program unit:
* revision 1.1 - 4/29/87 - added FRAME and REPEAT commands
* revision 1.2 - 7/7/87  - added PLOT command for line plots
* revision 1.3 - 9/9/87  - added LOAD command
* revision 1.4 - 9/23/87  - added DEFINE command
* revision 2.0 - 10/23/87 - changed DATA stmt rev. level to 2.00
*			    and added EXTERNAL statement
* revision 2.1 - 3/25/88 - added STATISTICS command
* revision 2.2 - 4/20/88 - ADDED &UNITS AND &DSET_PARMS TO EXTERNALS 2.02-->2.10
* revision 2.3 - 6/10/88 - renamed to Program FERRET
* revision 2.4 -  8/2/88 - for FERRET 1.10 - added SHADE and SPAWN commands
* revision 2.5 - 12/12/88- eliminated XGRID_DATA to new TMAP library
* V200:  1/21/90 - added user-definable command USER
*	  5/2/90 - added externals for BOX_BOUNDS and CK_GKS_ACTIVE which are
*		   called from PLOT+ routines
* Unix/RISC port:  pulled xrevision information fully into main program
*        7/12/91 - added note not to distribute
*       10/17/91 - added external for viewport sizing
* V230:   6/8/92 - added WIRE command
* V300      1/93 - new version (includes animations, etc.)
*                  added xgt_grid_data - forced from xcalc_vars_data.F by
*                  DEC f77 3.2
*           6/93 - added QUERY command, and EXTERNAL XGUI_DATA
* These changes happened concurrent to the restructure for V320:
* V311: 5/17/94 added FILE/ORDER and FILE/FORMAT=STREAM
* V312: 6/17/94 metafile bug fix in XGKS (hangs and crashes with color plots)

* as FERRET_DISPATCH routine (formerly MAIN routine ferret.F):
* V320: 5/19/94  (also removed xrevision.cmn DATA stmts to xrevision_data.F
*		  for AIX compatibility)
* 	2/28/95 - error message buffer for GUI moved to TMAP COMMON
*		- send "synch" message to GUI if new window is created
* V400: 3/20/95 - pass "memory" to get_command for preliminary pass in which
*		  grave accent expressions are evaluated
*	6/20/95 - send "synch" for window destruction, too
* V411: 9/13/95 - Ultrix port:  CHAR(integer*1) is illegal although CHAR(byte)
*		  is allowed ... go figger
* V420: 4/23/96 - added new commands IF, ELSE, ELIF, ENDIF
* Linux Port *kob* 1/97 - Added external for XWINDOW_STATE_DATA block data
* V450: 7/7/97 - added external for XGRID_CHG_FCNS block data
* V500: 2/99 *sh* - added POLYGON command
* V530: 10/00 *sh* - added external declaration for new xez_info_data
* V554:  9/03 *acm*- Loop 2 ran to 200; change to cmnd_buff_len 
*                    (came up when implementing the -script command-line option)

#ifdef unix
	include 'tmap_dims.parm'
	include 'xalt_messages.cmn_text'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'gui.parm'
	include 'command.parm'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
	include 'xgui.cmn'
	include 'xplot_state.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XALT_MESSAGES.CMN_TEXT'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:GUI.PARM'
	INCLUDE 'FERRET_CMN:COMMAND.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XGUI.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
#endif

* EXTERNAL references to force the LINKER to look for BLOCK DATA SUBROUTINES
	EXTERNAL
     .		XALGEBRA_DATA,
     .		XCOMMAND_DATA,
     .		XCONTEXT_DATA,
     .		XCONTROL_DATA,
     .		XERRMSG_TEXT_DATA,
     .		XGFDL_MASKS_DATA,
     .		XONEDIM_DATA,
     .		XPLOT_SETUP_DATA,
     .		XPLOT_STATE_DATA,
     .		XPROG_STATE_DATA,
     .		XTEXT_INFO_DATA,
     .		XVARIABLES_DATA,
     .		XCALC_VARS_DATA,
     .		XDSET_PARMS_DATA,
     .		XFR_GRID_DATA,
     .		XGKS_DEVICES_DATA,
     .		XGUI_DATA,
     .		XREVISION_DATA

        EXTERNAL XGT_GRID_DATA      ! 2/93

	EXTERNAL BOX_BOUNDS, CK_GKS_ACTIVE, GET_VIEW_SIZE,
     .           FERRET_PLOT_COMPLETE

* kob 1/97
	EXTERNAL XWINDOW_STATE_DATA
* sh 7/97
	EXTERNAL XGRID_CHG_FCNS_DATA,
     .		 XEZ_INFO_DATA

* temporarily we simply declare the "memory" array right here
* calling argument declarations
	INTEGER nflags, nchars, rtn_flags(nflags), nerrlines
	REAL memory(*)
#ifdef NEED_BYTE
        BYTE       command(*), rtn_chars(nchars)
#else
#   ifdef ultrix
        BYTE       command(*), rtn_chars(nchars)	! 9/95 Ultrix, only
#   else
        INTEGER*1  command(*), rtn_chars(nchars)
#   endif
#endif

* local variable declarations:
	INTEGER	TM_LENSTR1,
     .		cmnd_stack_level, reconfig, status, i, gui_action,
     .		insert_pt

* local parameter declarations
	INTEGER		slash_msg_continue
	PARAMETER     (	slash_msg_continue	  = 1)

* transfer the c-style string into a FORTRAN string
	cmnd_buff = ' '
c	DO 2 i = 1, 200
	DO 2 i = 1, cmnd_buff_len
	   IF ( command(i) .EQ. 0 ) GOTO 5
 2	cmnd_buff(i:i) = CHAR( command(i) )

 5	CALL GET_COMMAND( memory, cmnd_buff, status, *300 )

	GOTO (	 10, 20, 30, 40, 50, 60, 70, 80, 90,100,
     .		110,120,130,140,150,160,170,180,190,200,
     .          210,220,230,240,250,260,270 ) cmnd_num

 10	CALL XEQ_SET( memory, reconfig )
	IF ( reconfig .NE. 0 ) GOTO 400
	GOTO 300
 20	CALL XEQ_SHOW( memory )
	GOTO 300
 30	CALL XEQ_CANCEL( memory )
	GOTO 300
 40	CALL XEQ_CONTOUR( memory )	! after rev 1.00 1/5/87
	GOTO 300
 50	CALL XEQ_LIST( memory )
	GOTO 300
 60	CALL XEQ_PLOT( memory )
	GOTO 300
 70	CALL XEQ_GO
	GOTO 300
 80	CALL XEQ_HELP
	GOTO 300
 90	CALL XEQ_LOAD( memory )
	GOTO 300
 100	CALL XEQ_DEFINE( memory )
	GOTO 300
 110	CALL XEQ_EXIT( *500 )
	GOTO 300			! command file xit or error
 120	CALL XEQ_MESSAGE
	GOTO 300
 130	CALL XEQ_VECTOR( memory )
	GOTO 300
 140	CALL XEQ_PPLUS
	GOTO 300
 150	CALL XEQ_FRAME( memory )
	GOTO 300
 160	CALL XEQ_REPEAT
	GOTO 300
 170	CALL XEQ_STAT( memory )
	GOTO 300
 180	CALL XEQ_SHADE( memory )
	GOTO 300
 190	CALL XEQ_SPAWN
	GOTO 300
 200	CALL XEQ_USER_COMMAND( memory )
	GOTO 300
 210	CALL XEQ_WIRE( memory )
	GOTO 300
 220	CALL XEQ_QUERY
	GOTO 300
 230	CALL XEQ_IF
	GOTO 300
 240	CALL XEQ_ELSE
	GOTO 300
 250	CALL XEQ_ELIF
	GOTO 300
 260	CALL XEQ_ENDIF
	GOTO 300
 270	CALL XEQ_POLYGON( memory )
	GOTO 300

* finished a single command
 300	gui_action = factn_no_action	! default
	IF (    (cmnd_num    .EQ. cmnd_set
     .	   .AND. subcmnd_num .EQ. subcmnd_set_data)
     . .OR.     (cmnd_num    .EQ. cmnd_cancel
     .	   .AND. subcmnd_num .EQ. subcmnd_canc_data) ) THEN
* ... need to synchronize data sets with GUI
	   gui_action = factn_synch_set_data
	ELSEIF ( cmnd_num    .EQ. cmnd_message
     .	   .AND. qual_given(slash_msg_continue) .EQ. 0 ) THEN
	   gui_action = factn_pause
	ELSE
* ... need to synchronize output window creation/destruction with GUI
* ... creation may be from any (initial) plot command or from SET WINDOW
	   DO 302 i = 1, max_windows
	      IF ( wn_open(i) .NEQV. gui_window_reported(i) ) THEN
	         gui_action = factn_synch_window
	         rtn_flags(frtn_idata1)  = i
	         IF (wn_open(i)) THEN 
	            rtn_flags(frtn_idata2)  = 1   ! 1 means "creation"
	            gui_window_reported(i) = .TRUE.
	         ELSE
	            rtn_flags(frtn_idata2)  = -1  ! -1 means "destruction"
	            gui_window_reported(i) = .FALSE.
	         ENDIF
	      ENDIF	         
 302	   CONTINUE
	ENDIF

* ... prepare error buffer to pass back to calling (GUI?) program
	IF ( mode_gui .AND. gui_status .NE. ferr_ok ) THEN
	   gui_action = factn_display_error
	   insert_pt = 1
	   DO 310 i = 1, alt_nlines
	      nerrlines = i
	      CALL CSTRING_INSERT(alt_text(i), TM_LENSTR1(alt_text(i)),
     .				  rtn_chars, nchars, insert_pt )
	      IF ( insert_pt .GT. nchars ) GOTO 320
 310	   CONTINUE
	ELSE
	   nerrlines = 0
	ENDIF
 320	alt_nlines = 0
	CALL CLEANUP_LAST_CMND( cmnd_stack_level )
	IF ( .NOT.mode_gui ) THEN
	   GOTO 5
	ELSEIF ( cmnd_stack_level .GT. 0 ) THEN
	   rtn_flags(frtn_control) = fctrl_in_ferret  ! command not yet finished
	   rtn_flags(frtn_status)  = gui_status
	   rtn_flags(frtn_action)  = gui_action
	   RETURN
	ELSE
	   rtn_flags(frtn_control) = fctrl_back_to_gui
	   rtn_flags(frtn_status)  = gui_status
	   rtn_flags(frtn_action)  = gui_action
	   RETURN
	ENDIF
	   
* reconfigure main memory
 400	rtn_flags(frtn_control) = fctrl_in_ferret
	rtn_flags(frtn_status)  = gui_status
	rtn_flags(frtn_action)  = factn_mem_reconfigure
	rtn_flags(frtn_idata1)  = reconfig
	RETURN

* exit the program
 500	rtn_flags(frtn_control) = fctrl_back_to_gui
	rtn_flags(frtn_status)  = gui_status
	rtn_flags(frtn_action)  = factn_exit
	RETURN

	END
