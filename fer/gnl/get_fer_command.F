	SUBROUTINE GET_FER_COMMAND( called_string, status, * )

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
* renamed to GET_FER_COMMAND for gfortran build, which has a call, "GET_COMMAND"
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

* v500: 5/24/99 *kob* - Have to insert call to NAG F90_UNIX module in
*                       order to use the Fortan Flush.  USE command has
*                       to come right after subroutine name
*     : 5/34/99 *kob* - call FLUSH for all systems
* V530: *sh* 1/01 - suppress verify output during skipped IF lines 
*       *sh* 3/01 - do not call PARSE_COMMA_LIST if arg1 quoted
* V531: *sh* 4/01 - chenge prompt for continuation line
*	*sh* 5/01 - ignore DOS carriage returns in journal files
* *kob* 2/03 - replace call to intrinsic func CHAR with its octal
*              constant equivalent - needed for g77 port
* *kob* 4/06 - back off my_readline call - needed for 64-bit port
* v603 *acm* 5/07 Housekeeping: initialize arg_start and arg_end
*                 before call to parse_command
* V6.7 *kms* 8/11 add check for one-command mode before prompting
*                 user for the next command
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V690 *sh* 2/14 initialize max_context, now a variable to support aux vars
* V693+ *sh* 12/14 moved arg_start/end inits into PARSE_COMMAND (housekeeping)
*                 Moved other house-keeping from PARSE_COMMAND here
*	            1) Trim leading "yes? " and "...? "
*                   2) Trim bang-comments off.
*                   3) Do not call PARSE_COMMAND for pure comments.
*		    4) Process continuation lines
*                 Issue warning if GO file EOF found while in the midst of IF
* V697 12/15 *acm* Ticket 2337 Increase length of cmnd buffer to 20480
* V697 12/15 *acm* Ticket 2341 Check that cmnd_len hasn't been set to 0.
* V697 2/16 *acm* DO NOT Increase length of cmnd buffer to 20480 but keep 
*                 other related changes when reporting errors in long command lines
* V7+  6/16 *acm* Ticket 2446: Include the script name in warnings about 
*                 unclosed IF blocks
* V75+ 4/19 *acm* Issue 1927 (and 228!) initialize qual_given before parsing.
** 10/2022 *acm* Code cleanup: remove references to obsolete gui interface
* 12/2022 *acm* Code cleanup: Replace the computed GOTO statement with IF blocks
*                  GOTO (100,200,300) input_source

        IMPLICIT NONE

* calling argument declarations:
	INTEGER		status
	CHARACTER*(*)	called_string

* internal variable declarations:
	LOGICAL re_verify, do_verify, arg1_quoted,
     .		pure_comment, appending_cmnd 
	INTEGER	TM_LENSTR, TM_LENSTR1,
     .		iqual, ilist, cmnd_len, error_count,
     .          has_more, i, cmnd_len_w_cmnt, i1, i2
	CHARACTER prefix*5, raw_cmnd_buff*2048
	CHARACTER symbuff*32, scriptname*512

* local parameter declarations
        CHARACTER*1 backslash, tab
	CHARACTER*5 dprefix
	PARAMETER ( dprefix = ' !-> ') 
        PARAMETER ( backslash = char(92) )
	PARAMETER ( tab = CHAR(9) )

	include 'tmap_dims.parm'
	include 'errmsg.parm'
	include 'ferret.parm'
	include 'command.parm'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xcontrol.cmn'
	include 'xinterrupt.cmn'
	include 'xcontext.cmn'

* initialize
	error_count = 0
	max_context = max_noframe_context  ! 2/2014
	len_cmnd_copy = 0
	appending_cmnd = .FALSE.
        do_verify = .FALSE.


* initialize the qualifiers -- otherwise REPL_EXPRNS tries to take
* quals from previous commands into account. See tickets 228, 1927 

        DO iqual = 1, max_qual_list
	   qual_given( iqual ) = 0
	ENDDO

* if a command was given then execute it immediately
	IF ( called_string .NE. ' ' ) THEN
	   cmnd_buff = called_string
	   cmnd_len_w_cmnt = TM_LENSTR( cmnd_buff )
	   IF (   mode_journal
     .	    .AND. jrnl_lun .NE. unspecified_int4
     .	    .AND. cmnd_len_w_cmnt .GT. 0
     .	    .AND. cmnd_buff(1:1) .NE. backslash ) then
	      WRITE (jrnl_lun,'(A)') cmnd_buff(1:cmnd_len_w_cmnt)

* *kob* 5/99 - Flush buffer
	      CALL FLUSH(jrnl_lun)

	   ENDIF
	   GOTO 1000
	ENDIF

* select source for input
* If in one-command mode but asking to prompt user for the next
* command, then we got here from reaching the end of a go file
* or a multi-command line.  So return to the first alternative
* since the user is never prompted by ferret in one-command mode.
 20     IF ( one_cmnd_mode .AND. (input_source .EQ. 1) ) THEN
           RETURN 1
        ENDIF


************** OPERATOR AS SOURCE OF COMMAND LINE ****************************

      IF (input_source .EQ. 1) THEN

* operator sitting at their keyboard

	error_count = error_count + 1
	IF (error_count .GT. 2) THEN
* orderly shutdown on multiple errors
              cmnd_buff = 'EXIT /PROGRAM'
	ELSE IF ( appending_cmnd ) THEN
              CALL TM_FRIENDLY_READ('...? ',cmnd_buff,*20)
	ELSE
              CALL TM_FRIENDLY_READ(prompt(:prompt_len),cmnd_buff,*20)
	ENDIF
* clear interrupted if Ctrl-C was entered while typing command
	INTERRUPTED = .FALSE.
	cmnd_len_w_cmnt = TM_LENSTR( cmnd_buff )
	error_count = 0      ! in case comment or blank line returns here

* echo command line to journal file

	IF (   mode_journal
     .	 .AND. jrnl_lun .NE. unspecified_int4
     .	 .AND. cmnd_len_w_cmnt .GT. 0
     .	 .AND. cmnd_buff(1:1) .NE. backslash ) then
	   WRITE (jrnl_lun,'(A)') cmnd_buff(1:cmnd_len_w_cmnt)

* *kob* 5/99 - Flush buffer
	   CALL FLUSH(jrnl_lun)
	      
	ENDIF
	do_verify = .FALSE.

*************** COMMAND FILE AS SOURCE OF COMMAND LINE ***********************

      ELSE IF (input_source .EQ. 2) THEN

	READ  (cmnd_lun, '(A)', END=210) cmnd_buff

	cmnd_len_w_cmnt = TM_LENSTR1( cmnd_buff )
	IF (cmnd_buff(cmnd_len_w_cmnt:cmnd_len_w_cmnt) .EQ. pCR ) THEN
	    cmnd_buff(cmnd_len_w_cmnt:cmnd_len_w_cmnt) = ' '
	    IF (cmnd_len_w_cmnt .GT. 1) cmnd_len_w_cmnt = 
     .					            cmnd_len_w_cmnt - 1
	ENDIF
        IF ( cmnd_buff(1:1) .EQ. backslash ) THEN
           cmnd_buff(1:1) = ' '
	   do_verify = .FALSE.
        ELSE
	   do_verify = mode_verify
        ENDIF
	prefix = ' '
	GOTO 1000
* end of command file - return to next lower control stack level
* if we were in the midst of an open IF block, issue a warning
* Include the script name in the note.
 210	IF (if_conditional) THEN
	   IF (csp .LE. if_cs(ifstk) ) THEN
	      symbuff = 'GO_FILE'
	      CALL GETSYM( symbuff, scriptname, i2, i1 )
	      IF (i1 .EQ. 0) THEN
	         CALL WARN('Unclosed IF statement ... assuming ENDIF, in script '//scriptname(:i2) )
	      ELSE
	         CALL WARN('Unclosed IF statement ... assuming ENDIF')
	      ENDIF
	   ENDIF
	ENDIF
	CALL POP_CS
*          Consider an update of the graphics
           CALL FGD_CONSIDER_UPDATE(.FALSE.)
	   GOTO 20

*************** COMMAND STACK AS SOURCE OF COMMAND LINE ***********************

      ELSE IF (input_source .EQ. 3) THEN
      
	CALL CS_COMMAND( cmnd_buff, *20 )
	cmnd_len_w_cmnt = TM_LENSTR1( cmnd_buff )
	do_verify =  mode_verify .AND. mode_arg(pmode_verify,1).GE.2
	prefix = dprefix

      ENDIF

*********** SAVE THE RAW COMMAND TO USE FOR ECHOING *************
 1000	raw_cmnd_buff = cmnd_buff

*********** TRIM "yes?" and "...?" FROM START OF LINE
	DO WHILE (cmnd_buff(1:5) .EQ. 'yes? '
     .       .OR. cmnd_buff(1:5) .EQ. '...? ' )
	   IF ( cmnd_len_w_cmnt .LT. 6 ) THEN
	     cmnd_buff = ' '
	     cmnd_len_w_cmnt = 1
	   ELSE
	     cmnd_buff = cmnd_buff(6:cmnd_len_w_cmnt)
	     cmnd_len_w_cmnt = cmnd_len_w_cmnt - 5
	   ENDIF
	ENDDO

*********** TRIM COMMAND BUFF TO EXCLUDE BANG_COMMENTS  ********************
* find length of command line ("!" is a valid terminator to permit comments)
	cmnd_len = cmnd_len_w_cmnt   ! dflt len if no comment found
	pure_comment = .FALSE.
	IF (cmnd_len .LE. 0) pure_comment = .TRUE.
	IF (cmnd_buff(1:1) .EQ. '!'
     . .OR. cmnd_buff(1:1) .EQ. '*'
     . .OR. cmnd_buff(1:1) .EQ. ' ' .AND. cmnd_len .EQ. 1) THEN
	   pure_comment = .TRUE.
	ELSE
* ... search forward in the command string for the first non-escaped "!"
	   DO i = 2, cmnd_len_w_cmnt
	      IF  ( cmnd_buff(i  :i  ) .EQ. '!' 
     .	      .AND. cmnd_buff(i-1:i-1) .NE. backslash ) GOTO 1005
	   ENDDO
* ... no bang comment found
	   GOTO 1010
* ... found "!".  Now search backwards for the last non-white character
 1005	   pure_comment = .TRUE. ! default until non-white char found
           DO cmnd_len = i-1, 1, -1
              IF (  cmnd_buff(cmnd_len:cmnd_len) .NE. ' '
     .        .AND. cmnd_buff(cmnd_len:cmnd_len) .NE. tab ) THEN
	         pure_comment = .FALSE.
	         GOTO 1010
	      ENDIF
	   ENDDO
* ... nothing but blanks preceding the comment bang
	   pure_comment = .TRUE.
	ENDIF

*********** ECHO THE COMMAND WITH ITS COMMENTS ***********************
 1010	IF (.NOT.if_conditional .AND. do_verify ) THEN
* ... echoing in IF blocks is handled by PROCESS_IF
	   IF (prefix .EQ. ' ') THEN
* ... separate SPLIT_LIST call to avoid single char ' ' prefix
	      CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .				 cmnd_buff, cmnd_len_w_cmnt)
	   ELSE
	      CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .			prefix//cmnd_buff, cmnd_len_w_cmnt+5 )
	   ENDIF
	ENDIF

*********** IF WE ONLY HAVE A COMMENT GET A FRESH STRING *************
	IF (pure_comment .AND. .NOT.appending_cmnd) GOTO 20

*********** WIPE OUT COMMENT AND LEFT JUSTIFY  ********************
	IF (pure_comment) THEN
	   cmnd_buff = ' '
	   cmnd_len = 1
	ELSEIF (cmnd_len_w_cmnt .GT. cmnd_len) THEN
	   cmnd_buff(cmnd_len+1:cmnd_len_w_cmnt) = ' '
	ENDIF
	IF ( .NOT.appending_cmnd ) THEN
	   CALL LEFT_JUST( cmnd_buff, cmnd_buff, cmnd_len )
           IF ( cmnd_buff(1:1) .EQ. '?' ) THEN   ! "?" should be an ALIAS!!
	      cmnd_buff = 'HELP'    ! 8/92
	      cmnd_len = 4
	   ENDIF
	ENDIF

********** INCOMPLETE (CONTINUED) CMND LINE -- MUST GET MORE ***************
	IF (cmnd_buff(cmnd_len:cmnd_len) .EQ. backslash) THEN
* ... save this line as a partial command
	   IF (appending_cmnd) THEN
	      cmnd_copy = cmnd_copy(:len_cmnd_copy) // cmnd_buff(:cmnd_len)
	      len_cmnd_copy = len_cmnd_copy + cmnd_len
	      IF (len_cmnd_copy .GT. cmnd_buff_len) GOTO 5005
	   ELSE
	      cmnd_copy = cmnd_buff(:cmnd_len)
	      len_cmnd_copy = cmnd_len
	   ENDIF
* ... wipe out the backslash in the stored copy
	   cmnd_copy(len_cmnd_copy:len_cmnd_copy) = ' '
	   len_cmnd_copy = len_cmnd_copy - 1
* ... go back for more
	   appending_cmnd = .TRUE.
	   GOTO 20
	ENDIF

*************** ASSEMBLE COMPLETED CONTINUATION LINES ********************
	IF (appending_cmnd) THEN
	   IF ( len_cmnd_copy+cmnd_len .GE. cmnd_buff_len ) GOTO 5005 
	   cmnd_buff = cmnd_copy(:len_cmnd_copy) // cmnd_buff(:cmnd_len)
	   cmnd_len = len_cmnd_copy + cmnd_len
	   appending_cmnd = .FALSE.
	ENDIF

*********** WHEN WITHIN AN IF BLOCK, COMMANDS MAY BE SKIPPED ****************
	IF (if_conditional) CALL PROCESS_IF(cmnd_buff, cmnd_len,
     .			            do_verify, prefix, raw_cmnd_buff, *20)

*************** PARSE THE COMMAND LINE ***************************************
* 1/14 the command string now has no continuations and no comments
	CALL PARSE_COMMAND (  cmnd_buff,
     .			     max_arg_list, max_qual_list,
     .			     len_cmnd, cmnd_num, subcmnd_num, 
     .			     num_qualifiers, qualifier_list,
     .			     qual_start, qual_end,
     .			     num_args, arg_start, arg_end,
     .			     err_lun, re_verify, arg1_quoted, status	)
	IF ( status   .NE. ferr_ok ) RETURN 1	! messages already displayed
	IF ( cmnd_num .EQ. 0 ) GOTO 20
	IF ( re_verify ) THEN
	   cmnd_len = TM_LENSTR1( cmnd_buff )
	   IF ( mode_verify ) CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .					' !-> '//cmnd_buff, cmnd_len+5 )
	ENDIF

*************** PROCESS SEMICOLON-SEPARATED COMMAND GROUPS ********************
        IF ( cmnd_num .EQ. cmnd_semicolon )
     .       CALL CS_SEMICOLON( cmnd_buff, *20)

*************** REORGANIZE LIST OF QUALIFIERS GIVEN TO INTERNAL ORDER   ********

	DO iqual = 1, max_qual_list
	   qual_given( iqual ) = 0
	ENDDO
	DO ilist = 1, num_qualifiers
	   IF ( cmnd_buff( qual_start(ilist):qual_start(ilist) ) .NE. '@' )
     .				qual_given( qualifier_list( ilist ) ) = ilist
	ENDDO

*************** BREAK APART ITEMS SEPARATED BY COMMAS IN FIRST ARGUMENT ********

	IF ( num_args .GT. 0 ) THEN
	   IF ( arg1_quoted ) THEN
	      num_items = 1
	      item_start(1) = arg_start(1)
	      item_end(1)   = arg_end(1)
	   ELSE
	      CALL PARSE_COMMA_LIST
     .		      ( cmnd_buff, arg_start(1), arg_end(1), max_item_list,
     .			num_items, item_start, item_end, status )
	      IF ( status .NE. ferr_ok ) RETURN 1
	   ENDIF
	ELSE
	   num_items = 0
	ENDIF
	
	RETURN

* error exits
 5005	CALL ERRMSG( ferr_syntax, status,
     .			'command line too long -- exceeds 2048', *9000 )
! 2048 must match cmnd_buff_len
 9000	RETURN 1


	END
