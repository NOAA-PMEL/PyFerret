	SUBROUTINE GET_COMMAND( memory, called_string, status, * )


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
* get next command line and decode it placing information in COMMON
* source of command may be calling argument string, operator, command file,
* or internal macro command (e.g. REPEAT or semicolon command group)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/26/86
* revision 0.1 - 3/4/87  - changed /star to /@
* revision 1.0 - 4/30/87 - allowed for control by CS (control stack)
* revision 1.1 - 8/11/87 - changer ^Z treatment from keyboard
* revision 1.2 - 9/14/87 - added SMG_ routines for VMS line recall
* revision 1.3 - 11/30/87- interrupt handling
* revision 1.4 - 2/25/88 - corrected bug in EOF on non-keyboard input
* revision 1.5 - 1/24/89 - allow for interrupts of PPLUS operations
* revision 1.6 - 2/2/89  - check to see if journal file exists
* V200:   2/8/90 - left justify text
* Unix/RISC port - 1/91 - cant use "/star" in documentation
*			- also ifdef'ed the VMS line recall routines
* V230:  1/27/92 - replaced double ^D exit with Unix-like "q" (QUIT)
*        8/26/92 - interpret command beginning with "?" as "HELP"
* V300:   2/2/93 - implemented semicolon-separated command groups
*	  5/6/93 - implemented "$" command arguments
*	 6/16/93 - added mode_gui input
* V301:  2/24/94 - small change due to mode_arg as a 2D variable
*	 3/14/94 - under mode refresh (xwindows) tweak the X event loop before 
*		   each new command (important under GO control)
* V313: 31/10/94 - on AIX systems the output jrnl_lun buffer must be flushed.
*			... dont know why ...
* V320:  5/19/94 - accept calling argument string (from GUI control program)
*	 	   cleanup from last command removed to CLEANUP_LAST_CMND
* 	10/24/94 - abnormal return when command file finishes in mode_gui
*       12/29/94 - use SPLIT_LIST to redirect tty output for GUI
*	 1/18/95 - added MODE VERIFY:ALWAYS
*		 - and double EOF from the command line is a QUIT (else
*			redirected command files go into a loop)
*        2/24/95 - send GUI output to journal fiie
* V400: 3/20/95 - pass "memory" to parse_command for preliminary pass in which
*		  grave accent expressions are evaluated
* V420:  3/8/96 - add FLUSH on Solaris to keep the journal file up to date
*	4/28/96 - call PROCESS_IF for multi-line IF's

* calling argument declarations:
	INTEGER		status
	REAL		memory(*)
	CHARACTER*(*)	called_string

* internal variable declarations:
	LOGICAL reverify
	INTEGER	TM_LENSTR, TM_LENSTR1,
     .		iqual, ilist, cmnd_len, double_eof

        CHARACTER*1 backslash
        PARAMETER ( backslash = CHAR(92) )

#ifdef unix
	include 'errmsg.parm'
	include 'ferret.parm'
	include 'command.parm'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xcontrol.cmn'
	include 'xinterrupt.cmn'
	include 'xgui.cmn'
#else
	INTEGER	SMG$READ_COMPOSED_LINE, vax_code

	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:COMMAND.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
	INCLUDE 'FERRET_CMN:XCONTROL.CMN'
	INCLUDE 'FERRET_CMN:XINTERRUPT.CMN'
	INCLUDE 'FERRET_CMN:XGUI.CMN'
	INCLUDE '($SMGDEF)'
	EXTERNAL SMG$_EOF	! those losers removed it from _SMGDEF *MR*
#endif

* initialize
	gui_status = ferr_ok	! start w/ a clean bill o' health
	double_eof = 0

* if a command was given then execute it immediately
	IF ( called_string .NE. ' ' ) THEN
	   cmnd_buff = called_string
	   cmnd_len = TM_LENSTR( cmnd_buff )
	   IF (   mode_journal
     .	    .AND. jrnl_lun .NE. unspecified_int4
     .	    .AND. cmnd_len .GT. 0
     .	    .AND. cmnd_buff(1:1) .NE. backslash ) then
	      WRITE (jrnl_lun,'(A)') cmnd_buff(1:cmnd_len)
#ifdef AIX_XLF
	      CALL FLUSH(jrnl_lun)
#endif
#ifdef solaris
	      CALL FLUSH(jrnl_lun)
#endif
	   ENDIF
	   GOTO 1000
	ENDIF

* select source for input
 20	GOTO (100,200,300) input_source

************** OPERATOR AS SOURCE OF COMMAND LINE ****************************

 100	IF (mode_gui) THEN

* operator using mouse input device
	   CALL MOUSE_COMMAND (cmnd_buff)
	   cmnd_len = TM_LENSTR( cmnd_buff )

	ELSE
* operator sitting at his keyboard
!	   WRITE (ttout_lun,3000) prompt(1:prompt_len)
! 3000	   FORMAT (1X,A,$)	! vms
! 3000	   FORMAT (A,$)         ! unix
!	   READ  ( ttin_lun,'(A)',END=2000 ) cmnd_buff
!	   cmnd_len = TM_LENSTR( cmnd_buff )

! check for multiple ^D's from the keyboard (or EOF on stdin redirect)
	   double_eof = double_eof + 1
	   IF (double_eof .GT. 2) STOP       ! 1/18/95

#ifdef unix
* ... gnu readline with a FORTRAN jacket
           CALL TM_FRIENDLY_READ(prompt(:prompt_len),cmnd_buff,*20)
	   cmnd_len = TM_LENSTR( cmnd_buff )
	   double_eof = 0	! in case comment or blank line returns here
#else
* ... special call to recall and allow editing of past commands
	   vax_code = SMG_READ_COMPOSED_LINE(	vms_kbd_id,
     .						vms_kbd_tbl,
     .						cmnd_buff,
     .						prompt(:prompt_len),
     .						cmnd_len )
	   IF ( vax_code .eq. %LOC(SMG$_EOF) ) GOTO 20
#endif
	ENDIF

* echo command line to journal file

	IF (   mode_journal
     .	 .AND. jrnl_lun .NE. unspecified_int4
     .	 .AND. cmnd_len .GT. 0
     .	 .AND. cmnd_buff(1:1) .NE. backslash ) then
	   WRITE (jrnl_lun,'(A)') cmnd_buff(1:cmnd_len)
#ifdef AIX_XLF
	   CALL FLUSH(jrnl_lun)
#endif
	endif
	GOTO 1000

*************** COMMAND FILE AS SOURCE OF COMMAND LINE ***********************

 200	READ  (cmnd_lun, '(A)', END=210) cmnd_buff
	cmnd_len = TM_LENSTR1( cmnd_buff )
        IF ( cmnd_buff(1:1) .EQ. backslash ) THEN
           cmnd_buff(1:1) = ' '
        ELSEIF ( mode_verify ) THEN
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, cmnd_buff, cmnd_len)
        ENDIF
	GOTO 1000
* end of command file - return to next lower control stack level
 210	CALL POP_CS
	IF ( csp.EQ.0 .AND. mode_gui ) THEN		! GUI code ...
	   gui_status = ferr_ok	! a clean bill o' health at GO termination
	   RETURN 1
	ELSE
	   GOTO 20
	ENDIF

*************** COMMAND STACK AS SOURCE OF COMMAND LINE ***********************

 300	CALL CS_COMMAND( cmnd_buff, *20 )
	cmnd_len = TM_LENSTR1( cmnd_buff )
	IF ( mode_verify .AND. mode_arg(pmode_verify,1).GE.2)
     .		CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .			' !-> '//cmnd_buff, cmnd_len+5 )

*********** LEFT JUSTIFY THE COMMAND  ********************
 1000	CALL LEFT_JUST( cmnd_buff, cmnd_buff, cmnd_len )
        IF ( cmnd_buff(1:1) .EQ. '?' ) cmnd_buff = 'HELP'    ! 8/92

*********** POSSIBLY SKIP COMMAND UNDER IF CONTROL (4/96) ********************
	IF (if_conditional) CALL PROCESS_IF(cmnd_buff, *20)

*************** DECODE THE COMMAND LINE ***************************************
	CALL PARSE_COMMAND ( memory, cmnd_buff,
     .			     max_arg_list, max_qual_list,
     .			     len_cmnd, cmnd_num, subcmnd_num, 
     .			     num_qualifiers, qualifier_list,
     .			     qual_start, qual_end,
     .			     num_args, arg_start, arg_end,
     .			     err_lun, reverify, status	)
	IF ( status   .NE. ferr_ok ) RETURN 1	! messages already displayed
	IF ( cmnd_num .EQ. 0 ) GOTO 20
	IF ( reverify ) THEN
	   cmnd_len = TM_LENSTR1( cmnd_buff )
	   IF ( mode_verify ) CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .					' !-> '//cmnd_buff, cmnd_len+5 )
	ENDIF

*************** PROCESS SEMICOLON-SEPARATED COMMAND GROUPS ********************
        IF ( cmnd_num .EQ. cmnd_semicolon )
     .       CALL CS_SEMICOLON( cmnd_buff, *20)

*************** REORGANIZE LIST OF QUALIFIERS GIVEN TO INTERNAL ORDER   ********

	DO 1010 iqual = 1, max_qual_list
 1010	qual_given( iqual ) = 0
	DO 1020 ilist = 1, num_qualifiers
	   IF ( cmnd_buff( qual_start(ilist):qual_start(ilist) ) .NE. '@' )
     .				qual_given( qualifier_list( ilist ) ) = ilist
 1020	CONTINUE

*************** BREAK APART ITEMS SEPARATED BY COMMAS IN FIRST ARGUMENT ********

	IF ( num_args .GT. 0 ) THEN
	   CALL PARSE_COMMA_LIST
     .		      ( cmnd_buff, arg_start(1), arg_end(1), max_item_list,
     .			num_items, item_start, item_end, status )
	   IF ( status .NE. ferr_ok ) RETURN 1
	ELSE
	   num_items = 0
	ENDIF
	
	RETURN

	END
