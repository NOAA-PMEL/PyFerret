	SUBROUTINE ERRMSG ( status_in, status_out, err_string, * )

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
* display explanatory message of error, transfer error code to output variable,
* and always take alternate return to serve as error branch in
* calling program
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/31/86 - heavily based on TM_ERRMSG
* revision 0.1 -  4/2/86 - changed FORMAT for ERR = messages
* revision 0.2 - 4/29/87 - depending on mode_ignore_err possibly return
*			   control of program to interactive mode
* revision 0.3 - 5/14/87 - saves "last_error"
* revision 0.4 - 11/30/87- handles interrupts
* revision 0.5 - 1/12/89 - prefix error messages with "**ERROR:"
*			 - eliminated non-standard "_" format
* V200:  6/21/89 - check to see if "HELP CUE" is requested
*	11/14/89 - special processing of Panasonic "ODR" errors
*	11/27/89 - multiple line messages
* Unix/RISC port - 1/91 - cant use "//" with CHAR*(*) variable
*			- different processing of ferr_erreq
*                 11/91 - "HELP" cues only relevant on VMS
* V230:  7/21/92 - improved formatting of Unix system messages using PERROR arg
*        11/4/92 - trim line end blanks from first line of message
*         2/3/93 - display command line if control stack is "up"
* V300:  4/30/93 - added ferr_silent (for GO arguments with own error text)
*	 7/93 - added changes for GUI-based control
* V301:	11/93 - the SUN f77 also has a bug with "//" on char*(*) variables.
*		It doesn't show up as a compiler error message - instead it
*		causes "RETURN 1" to do a normal return
* V320: 2/28/95 - SPLIT_GUI_MESSAGE --> TM_SPLIT_MESSAGE
* V420: 4/28/96 - close out all open IF blocks on error
* 12/96 - LINUX Port - *kob*
*		     - had to insert a "!" in column 1 of a purely comment line
*		     - had to remove a single quote from comment line - 
*			compiler did not like it...
*		      - added ifdef to use linux perror c routine rather than
*			fortran perror.  3.11.97
* *kob* 1/00 - v5.30 - add xplot_setup for animate logical.  if ctrl-c
*                      interrupt during animation, clean things up
* arguments:
*	status	- error code from calling program
*	status_out	- error code (echoed) back to calling program
*	err_string	- error text supplied from calling program - typically
*				a filename or variable name
* *acm 1/02  -v5.41  -- Check for mode ignore_error before closing out all open 
*                       IF blocks on error  (at label 1000)
* *acm* 10/04 v5.80  -- Check flag silent_err. This is set for SHO VAR/XML when
*                       checking for user variables not relevant to the context
* *acm* 10/04 v5.80  -- New code ferr_nomessge for use when user-given error msg 
*                       in coached string substitution
* *acm* 11/05 V600   -- New symbol FER_LAST_ERROR contains all the text normally
*                       written to std out. 
* *acm*  5/06 V600   -- Concatenate multiple lines for FER_LAST_ERROR with backslash.

	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'
	include 'xcontrol.cmn'
	include 'xerrmsg_text.cmn'
        include 'xrisc.cmn'
        include 'xgui.cmn'
	include 'xplot_setup.cmn'


* ^C handling routine
	EXTERNAL CTRLC_AST

* calling argument declarations:
	INTEGER		status_in,status_out
	CHARACTER*(*)	err_string

* internal variable declarations:
	LOGICAL 	help_cue, multiline
	INTEGER		TM_LENSTR1, len_text, len_string, status,
     .			cr_pos, ier, ist, len_symbuf
        CHARACTER*1028  symbuf  ! for symbol FER_LAST_ERROR

* see if "HELP CUE" was requested
	help_cue = status_in .GT. ghelp_cue
	IF ( help_cue ) THEN
	   status = status_in - ghelp_cue
	ELSE
	   status = status_in
	ENDIF

* initialize
	last_error = status			! save it for "HELP"
	status_out = status			! transfer error code to output
	gui_status = status

* initialize string for symbol FER_LAST_ERROR

         symbuf = ' '
         len_symbuf = 1

#ifdef unix
           risc_buff = err_string     ! DEC f772.1 cannot use // with char*(*)
! kob 12/96					! same for SUN 11/93
#endif

* display error message 
	IF ( status .GT. greport .AND. (.NOT.silent_err) ) THEN
	   len_text = TM_LENSTR1( err_text(status - greport) ) 
	   IF ( err_string .EQ. ' ' ) THEN
	      CALL TM_SPLIT_MESSAGE( err_lun, ' **ERROR: '//
     .				err_text(status-greport)(:len_text) )
     
              symbuf(len_symbuf:) = ' **ERROR: '//
     .				err_text(status-greport)(:len_text)
              len_symbuf = len_symbuf + len_text + 10

	   ELSE
* ... multiple lines ?
	      cr_pos = INDEX( err_string, pCR )
	      multiline = cr_pos .NE. 0
	      IF ( multiline ) THEN
	         len_string = TM_LENSTR1( err_string(:cr_pos-1) )  ! 11/92 mod
	      ELSE
	         len_string = TM_LENSTR1( err_string )
	      ENDIF

              IF (err_text(status-greport) .EQ. ' ') THEN
 	         CALL TM_SPLIT_MESSAGE( err_lun, ' **ERROR: '//
     .			        risc_buff(:len_string) )
!     .				': '//err_string(:len_string) )  !11/93 SUN bug

                 symbuf(len_symbuf:) = ' **ERROR: '//
     .			        risc_buff(:len_string) 
                 len_symbuf = len_symbuf + len_string + 10

              ELSE
                 ier = status
                 ier = greport
                 ier = status - greport
                 len_text = TM_LENSTR1(err_text(status-greport))
	         CALL TM_SPLIT_MESSAGE( err_lun, ' **ERROR: '//
     .				err_text(status-greport)(:len_text)//
     .				': '//risc_buff(:len_string) )
!     .				': '//err_string(:len_string) )  !11/93 SUN bug

                 symbuf(len_symbuf:) =' **ERROR: '//
     .				err_text(status-greport)(:len_text)//
     .				': '//risc_buff(:len_string) 
                 len_symbuf = len_symbuf + len_text + len_string + 13

              ENDIF
    
	      IF ( multiline ) THEN
                 CALL WRITE_MULTI(err_lun, err_string(cr_pos+1:))
                 symbuf(len_symbuf:len_symbuf) = '\\'
                 symbuf(len_symbuf+1:) = err_string(cr_pos+1:)
                 len_string = TM_LENSTR1(err_string(cr_pos+1:))
                 len_symbuf = len_symbuf + len_string + 2
                 DO WHILE (INDEX( symbuf, pCR ) .GT. 0)
                   cr_pos = INDEX( symbuf, pCR )
                   symbuf(cr_pos:cr_pos) = '\\'
                 ENDDO
              ENDIF

	   ENDIF
 3000	   FORMAT (' **ERROR: ',3A)

	ELSEIF ( status .EQ. ferr_ODR_error ) THEN
	   len_string = TM_LENSTR1( err_string )
	   WRITE (err_lun,3100) err_string(:len_string)

           symbuf(len_symbuf:) =err_string(:len_string)
           len_symbuf = len_symbuf + len_string + 1

 3100	   FORMAT (' **Panasonic error: ',A)

* error requiring no message ?
	ELSEIF ( status .GT. gnoreport ) THEN
	   GOTO 1000

* error message from ERR= branch ?
	ELSEIF ( status .EQ. ferr_erreq ) THEN

#ifdef unix
	   len_string = TM_LENSTR1( risc_buff )
#ifdef F90_SYSTEM_ERROR_CALLS
	   call linux_perror('**ERROR: '//risc_buff(:len_string) )
#else
           CALL PERROR( '**ERROR: '//risc_buff(:len_string) )
#endif
#else
	! code by Mark Renton
	! fetch the system error codes for most recent error

	   call ERRSNS ( fnum, rmssts, rmsstv, iunit, condval )

	   if ( ( rmssts .EQ. 0 ) .AND. ( rmsstv .EQ. 0 ) ) then

		msgvec ( 1 ) = 1
		msgvec ( 2 ) = condval
		msgvec ( 3 ) = 0
	   else
		msgvec ( 1 ) = 1
		msgvec ( 2 ) = rmssts
		if ( rmsstv .ne. 0 ) then
			msgvec ( 1 ) = 2
			msgvec ( 3 ) = rmsstv
		endif
	   endif

	   call sys_putmsg ( msgvec, , , )	! output the message

	   IF ( err_string .NE. ' ' ) THEN
              WRITE (err_lun,'(1X,A)') err_string
              
              len_string = TM_LENSTR1( err_string )
              symbuf(len_symbuf:) = err_string(:len_string)
              len_symbuf = len_symbuf + len_string + 1
#endif

* interrupt processing ? Re-store ^C interrupt handler
	ELSEIF ( status .EQ. ferr_interrupt ) THEN
	   CALL TM_SPLIT_MESSAGE( ttout_lun, '** INTERRUPTED! **')
	   CALL SET_CTRL_C( CTRLC_AST )
* *kob*	if animating during an interrupt, lets clean up
	   IF (animate .EQV. .TRUE.) THEN
	      animate = .FALSE.
	      CALL SWAP_ACTIVE_WINDOW
	   ENDIF
	ELSEIF ( status .EQ. ferr_interrupt ) THEN
* silent error code ?
	ELSEIF ( status .EQ. ferr_silent ) THEN
	   CONTINUE        ! quietly!

	ELSE
* unrecognized error code ?
	   CALL TM_SPLIT_MESSAGE( err_lun, 'unknown error code:')              
           symbuf(len_symbuf:) = 'unknown error code:'
           len_symbuf = len_symbuf + 19

	END IF

* close out all open IF blocks on error (4/96)
* but only if we're not ignoring errors (4/02)
 1000	IF ( .NOT.mode_ignore_err) then
           ifstk = 0
	   if_conditional = .FALSE.
        ENDIF

* if program was under internal control ( from the control stack ) this error
* indicates that control is to be returned to the operator
	IF ( cs_in_control ) THEN
* ... display the offending command
	   IF (status.NE.ferr_silent) CALL TM_SPLIT_MESSAGE( err_lun,
     .					cmnd_buff(1:len_cmnd) )
           IF ( status.EQ.ferr_interrupt
     .    .OR. .NOT.mode_ignore_err    ) THEN
* ... collapse the control stack back to simple operator control
 1100	      IF ( csp .GT. 0 ) THEN
	         CALL POP_CS
	         GOTO 1100
	      ENDIF
* ... inform the operator.  If running in -script mode, exit from Ferret.
	      CALL TM_SPLIT_MESSAGE( err_lun,
     .   'Command file, command group, or REPEAT execution aborted')
              IF (its_script) STOP '-script mode, ERROR RUNNING SCRIPT'  !exit the program
	   ENDIF
	ENDIF

* prompt with Help cue ?
#ifdef unix
* HELP command doesnt function on Unix systems
#else
	IF ( help_cue ) WRITE (err_lun,*) 'Type HELP error for assistance'
#endif

* Generate symbol containing the whole error message

* was there an external function error
         IF ( len_eftext .GT. 0) THEN
            symbuf(len_symbuf:) = '. Bailing out of external function '
     .         //efname(:len_efname)//'\\'
            len_symbuf = len_symbuf + len_efname + 36
            symbuf(len_symbuf:) = ef_errtext(:len_eftext)
            len_symbuf = len_symbuf + len_eftext
         ENDIF

* TMAP_error was saved in FER_LAST_ERROR already. Otherwise save
* the string in the symbol

         IF (status_in .NE. ferr_TMAP_error) THEN
            ier = 0
            ist = 0
            CALL SETSYM ('FER_LAST_ERROR '//symbuf, 14+len_symbuf, 
     .                 ier, ist)

         ENDIF

         len_eftext = 0

* take alternate RETURN always
  500	RETURN 1

	END
