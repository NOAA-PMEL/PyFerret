	SUBROUTINE FERRET_DISPATCH(  command, rtn_flags, nflags,
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
* V553:  9/03 *acm*- Loop 2 ran to 200; change to cmnd_buff_len 
*                    (came up when implementing the -script command-line option)
* V63  *acm* 10/09 Changes for gfortran build
* V664 *kms* 8/10 Add second alternate return from XEQ_EXIT to exit the dispatch
*                 loop but just re-enter the loop if not under pyferret
*       *acm* 3/12 removing use USER command
* V685 *acm* 11/13 Issue an ERROR not a NOTE if USER command is tried.
* V695 *acm*  6/16 New ANNOTATE command.
* V702 *sh*   2/17 dynamic memory management - no return for SET MEMORY
** 10/2022 *acm* Code cleanup: remove references to obsolete gui interface

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'command.parm'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
	include 'xplot_state.cmn'

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
     .		XFR_GRID_DATA,
     .		XGKS_DEVICES_DATA,
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
#ifdef NEED_BYTE
        BYTE       command(*), rtn_chars(nchars)
#else
        INTEGER*1  command(*), rtn_chars(nchars)
#endif

* local variable declarations:
	INTEGER	TM_LENSTR1,
     .		cmnd_stack_level, reconfig, status, i, insert_pt

* local parameter declarations
	INTEGER		slash_msg_continue
	PARAMETER     (	slash_msg_continue	  = 1)

* transfer the c-style string into a FORTRAN string
	cmnd_buff = ' '
	DO i = 1, cmnd_buff_len
	   IF ( command(i) .EQ. 0 ) GOTO 5
 	cmnd_buff(i:i) = CHAR( command(i) )
	ENDDO

 5	CALL GET_FER_COMMAND(  cmnd_buff, status, *300 )

	IF (cmnd_num .EQ. 1) THEN
	   CALL XEQ_SET
	ELSE IF (cmnd_num .EQ. 2) THEN
	   CALL XEQ_SHOW
	ELSE IF (cmnd_num .EQ. 3) THEN
	   CALL XEQ_CANCEL
	ELSE IF (cmnd_num .EQ.  4) THEN
	   CALL XEQ_CONTOUR
	ELSE IF (cmnd_num .EQ.  5) THEN
	   CALL XEQ_LIST
	ELSE IF (cmnd_num .EQ.  6) THEN
	   CALL XEQ_PLOT
	ELSE IF (cmnd_num .EQ.  7) THEN
	   CALL XEQ_GO
	ELSE IF (cmnd_num .EQ.  8) THEN
	   CALL XEQ_HELP
	ELSE IF (cmnd_num .EQ.  9) THEN
	   CALL XEQ_LOAD
	ELSE IF (cmnd_num .EQ. 10) THEN
	   CALL XEQ_DEFINE
	ELSE IF (cmnd_num .EQ. 11) THEN
	   CALL XEQ_EXIT( *500, *450 )
	ELSE IF (cmnd_num .EQ. 12) THEN
	   CALL XEQ_MESSAGE
	ELSE IF (cmnd_num .EQ. 13) THEN
	   CALL XEQ_VECTOR
	ELSE IF (cmnd_num .EQ. 14) THEN
	   CALL XEQ_PPLUS
	ELSE IF (cmnd_num .EQ. 15) THEN
	   CALL XEQ_FRAME
	ELSE IF (cmnd_num .EQ. 16) THEN
	   CALL XEQ_REPEAT
	ELSE IF (cmnd_num .EQ. 17) THEN
	   CALL XEQ_STAT
	ELSE IF (cmnd_num .EQ. 18) THEN
	   CALL XEQ_SHADE
	ELSE IF (cmnd_num .EQ. 19) THEN
	   CALL XEQ_SPAWN
	ELSE IF (cmnd_num .EQ. 20) THEN
	   CALL ERRMSG(ferr_syntax, status,
     .   'USER command not available in this Ferret version ', *300)
	ELSE IF (cmnd_num .EQ. 21) THEN
	   CALL XEQ_WIRE
	ELSE IF (cmnd_num .EQ. 22) THEN
	   CALL XEQ_QUERY
	ELSE IF (cmnd_num .EQ. 23) THEN
	   CALL XEQ_IF
	ELSE IF (cmnd_num .EQ. 24) THEN
	   CALL XEQ_ELSE
	ELSE IF (cmnd_num .EQ. 25) THEN
	   CALL XEQ_ELIF
	ELSE IF (cmnd_num .EQ. 26) THEN
	   CALL XEQ_ENDIF
	ELSE IF (cmnd_num .EQ. 27) THEN
	   CALL XEQ_POLYGON
	ELSE IF (cmnd_num .EQ. 28) THEN
	   CALL XEQ_ANNOTATE
	ENDIF

* finished a single command
 300	CONTINUE

* ... prepare error buffer to pass back to calling program
	nerrlines = 0
	CALL CLEANUP_LAST_CMND( cmnd_stack_level )
        IF ( one_cmnd_mode .AND. (cmnd_stack_level .EQ. 0) ) GOTO 450
	GOTO 5
	   
* exit the dispatch loop - if from PyFerret, return control to python
*                          if from Ferret, just re-enter the dispatch loop
 450    rtn_flags(1) = 2  ! fctrl_in_ferret   = 2
        rtn_flags(2) = ferr_ok
        rtn_flags(3) = 0  ! factn_no_action = 0
        RETURN

* exit the program
 500	rtn_flags(1) = 2  ! fctrl_in_ferret   = 2
	rtn_flags(2) = ferr_ok
	rtn_flags(3) = 2  ! factn_exit = 2
	RETURN

	END

*
* Routine to set the one_cmnd_mode in FERRET_DISPATCH 
* The following only applies if gui_mode is .FALSE.
* If new_mode_int is zero, FERRET_DISPATCH will return only 
*     on memory resizes and exit (original behavior)
* If new_mode_int is non-zero, FERRET_DISPATCH will return 
*     after each submitted command is complete
*
        SUBROUTINE SET_ONE_CMND_MODE(new_mode_int)

        IMPLICIT NONE
	include 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'

        INTEGER new_mode_int

        one_cmnd_mode = (new_mode_int .NE. 0)
        RETURN

        END

