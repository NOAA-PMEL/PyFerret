	SUBROUTINE XEQ_GO

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
* execute the GO command
* GO filename	indicates that the commands in file filename are to be executed

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/8/86
* revision 1.0 - 4/29/87 - added ability to nest GO commmands
*			 - eliminated pretenses of batch mode
* revision 1.1 - 7/22/88 - enhanced error message for no filename given
* Unix/RISC port - change OPEN statement for Unix
* V300: 4/93 - implemented arguments to GO commands (save command string here)
*	6/93 - bug fix to arguments when filename is in quotes
* V314 8/24/94 *kob* IBM port - READONLY not allowed in open statements on
*				IBM
*			      - SHARED not allowed in open statementson IBM..
*				add an ifdef NO_OPEN_SHARED
* V320: 10/94 - added GO/HELP filename
*	12/94 - added definition of PLOT+ symbol 'GO_FILE' if PLOT+ is running
*		and the GO command is given from the keyboard
*    12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* V411: 9/11/95 (IBM port) variable go_help assigned to integer expression
* V510 *sh* 3/00 -- increase allowed path name length
* V606  8/07 *acm* Send informational and error messages that are returned from 
*                  commands via SPLIT_LIST to std error rather than std out.
* V610  2/08 *acm* Always define a symbol LAST_GO_FILE with the name of the jnl script.
* V618  1/09 *acm* The last character of symbol LAST_GO_FILE was chopped off. Fix this.
* V66   *kms* 5/10 LIB_GET_LUN is a subroutine, not a function
* V687  *ACM* 3/14 for ticket 2044 write note !Enter_GO with script and arguments, only 
*                  if SET REDIRECT is turned on.
* V698 *sh* 3/16 - define GO_FILE always

! Note: A known "bug" of the 12/94 GO_FILE symbol is that the PLOT+ symbol
!	GO_FILE gets defined only **after** PLOT+ has already placed a window
!	on the screen and a GO command is executed.  This means that if the
!       GO file to create a plot contains the first command that opens
!       an output window the symbol GO_FILE will **not** be defined.
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)

        IMPLICIT NONE
	include 'tmap_dims.parm'
        include 'tmap_errors.parm'
	include 'ferret.parm'
	include 'command.parm'
	include 'errmsg.parm'
	include 'plot_setup.parm'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'		! 12/94
	include 'xcontrol.cmn'
	include 'xcontrol.equiv'
	include 'xrisc.cmn'

* internal variable declarations:
	LOGICAL	go_help
	INTEGER	TM_LENSTR1,
     .		temp_lun, status, TMAP_status,
     .          file_opened, ier, ist, len_string, slen
        CHARACTER TM_INQ_PATH*2048, cbuff*2048

* internal parameter declarations
	INTEGER     slash_go_help
	PARAMETER ( slash_go_help = 0 + 1 )
	
* was /HELP given ?
	go_help = qual_given( slash_go_help ) .GT. 0	! added ".GT. 0" 9/95

* check syntax
	IF     ( num_args .EQ. 0 ) THEN
	   IF (go_help) THEN
	      CALL SPLIT_LIST(pttmode_help, err_lun,
     .			' Use the GO command to name a file of FERRET'//
     .			' commands to be executed.', 0)
	      CALL SPLIT_LIST(pttmode_help, err_lun,
     .			 '     e.g.   yes? GO filename', 0)
	      CALL SPLIT_LIST(pttmode_help, err_lun, ' ', 1)
	      CALL SPLIT_LIST(pttmode_help, err_lun,
     .			' Use "GO/HELP filename" to read documentation in '//
     .			' the file to be executed.', 0)
	      RETURN
	   ELSE
	      CALL ERRMSG( ferr_unknown_arg, status,
     .			cmnd_buff(:len_cmnd)//': What file ?', *5000 )
	   ENDIF
	ENDIF

* check path list to find the "GO" file
        cbuff = TM_INQ_PATH( cmnd_buff(arg_start(1):arg_end(1)),
     .                   'FER_GO', '.jnl', .TRUE., TMAP_status )
        IF ( TMAP_status .NE. merr_ok ) GOTO 5100

* kob 1/2006 - need this after backing out of yonghua's changes 
*get a logical unit number and open the command file
	CALL LIB_GET_LUN( temp_lun )

* display HELP lines using the "more" command on the system
	IF (go_help) THEN
	   CALL SPLIT_LIST(pttmode_help, err_lun,
     .		' Full path name: '//cbuff(:TM_LENSTR1(cbuff)), 0) 

	   CALL SYSTEM( 'more -d '//cbuff(:TM_LENSTR1(cbuff)) )
	   RETURN
	ENDIF

* OPEN the file in preparation for reading commands from it
	OPEN(	UNIT=temp_lun,
     .		FILE=cbuff,
     .		STATUS='OLD',
     .		ERR=5010 )

* inform the control stack that it has taken control at a new level
	CALL STACK_PTR_UP( csp, cs_max, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	cs_in_control = .TRUE.
	cs_cmnd_num( csp ) = cmnd_go
	cs_lun( csp ) = temp_lun
	cmnd_lun = temp_lun
	input_source = input_from_file

* GO line will be parsed by DOLLAR_ARG - un-do parsing of PARSE_COMMAND
	CALL ALL_1_ARG
	cs_text( csp ) = cmnd_buff(arg_start(1):arg_end(1))
	
* ... Write note with script and arguments, only if SET REDIRECT is turned on.
        IF (redirect_stdout_flags .NE. redirect_none) THEN
	   risc_buff = '! Enter_GO '//cs_text(csp)(:TM_LENSTR1(cs_text(csp)) )
	   slen = 0
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, slen)
	ENDIF

	ier = 0
	ist = 0
	len_string = TM_LENSTR1( cbuff )

* define the symbol 'GO_FILE' -- it will always refer to the GO file currently
* in control.  Resets when the current script ends.
	cs_go_filename( csp) = cbuff   ! save the name to pop back later
	CALL SETSYM ('GO_FILE '//cbuff, 8+len_string, 
     .                 ier, ist)

* define the symbol 'LAST_GO_FILE' no matter what... -- it will refer to the
* last GO file that has been initiated.  When a GO file is initiated from
* within another and then comes to an end, LAST_GO_FILE will no longer be the
* one that is currently in control
	CALL SETSYM ('LAST_GO_FILE '//cbuff, 13+len_string, 
     .                 ier, ist)

* successful completion
	RETURN

* error exits
 5000	RETURN
 5010	CALL ERRMSG( ferr_erreq, status,
     .		cmnd_buff(arg_start(1):arg_end(1)), *5000)
 5100	CALL ERRMSG( ferr_TMAP_error, status, ' ', *5000)
 5200	CALL ERRMSG( ferr_out_of_range, status, 'GO/HELP=#-of_lines',
     .		*5000)
	END

