	SUBROUTINE XEQ_MESSAGE

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
* execute the MESSAGE command

* MESSAGE text of message	- displays message, prompts for <CR> and waits
* MESSAGE/QUIET text		- displays message and waits for <CR>
* MESSAGE/CONTINUE text		- displays message and continues

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/10/86
* V230:  7/30/92 - bug fix: avoid final quote ("hello" comes out 'hello"' now)
*        8/11/92 - allow blank messages (blank line on output)
*       12/15/92 - pause w/ TM_FRIENDLY_READ so SGI's w/out backing store work
*                - and fix non-functioning of /QUIET on Unix (was .NOT.integer)
* V300:  4/22/93 - fix treatment of enclosing quotes with ALL_1_ARG
*	 6/16/93 - different prompts, etc. for GUI control
*		 - interrupt if GUI interrupt sequence is received
* V314	8/25/94 *kob* IBM port - need to correct syntax error
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
*	 2/28/95 - SPLIT_GUI_MESSAGE --> TM_SPLIT_MESSAGE
* Linux Port - *kob* 12/96 - had to put single quotes around hello 
*			     in comment up above - linux f90 freaked out.
*       10.28.98 *js* Don't pause for messages if in server mode
* V510 5/00 *sh* - added MESSAGE/JOURNAL
* V520 6/00 *sh* - added MESSAGE/ERROR
* V650 3/10 *acm - added MESSAGE//OUTFILE= /APPEND /CLOBBER
** 10/2022 *acm* Code cleanup: remove references to obsolete gui interface
*                TM_SPLIT_MESSAGE replaced by SPLIT_LIST

        IMPLICIT NONE
	include 'ferret.parm'
	include 'tmap_dims.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'
        include 'xrisc.cmn'
	include 'xplot_state.cmn'

* local parameter declarations
	INTEGER		slash_continue, slash_quiet, slash_journal,
     .			slash_error, slash_file, slash_append, 
     .			slash_clobber
	PARAMETER     (	slash_continue	  = 1,
     .			slash_quiet	  = 2,
     .			slash_journal	  = 3,
     .			slash_error	  = 4,
     .			slash_file	  = 5,
     .			slash_append	  = 6,
     .			slash_clobber	  = 7)
	
* local variable declarations
	LOGICAl append, clobber
	INTEGER status, wsid, slen, rlen, sho_file
	PARAMETER (slen=1024)
	CHARACTER tmpfile*1024
#ifdef sun
        BYTE      ctmpfile(slen)
#else
        INTEGER*1 ctmpfile(slen)
#endif
	LOGICAL IS_SERVER

* treat all text as a single argument
	CALL ALL_1_ARG

* write the message if one has been supplied
	IF ( num_args .EQ. 1 ) THEN

* ... message to stderr
	   IF ( qual_given( slash_error ) .GT. 0 ) THEN
	      WRITE (err_lun, '(A)')
     .				cmnd_buff(arg_start(1):arg_end(1))
	      RETURN

* ... message to a named file
	   ELSEIF ( qual_given( slash_file ) .GT. 0 ) THEN
	      sho_file = qual_given( slash_file )
	      clobber = qual_given( slash_clobber ) .GT. 0
	      append = qual_given( slash_append ) .GT. 0

	      IF (sho_file .GT. 0) CALL OPEN_SHOW_FILE (show_lun, 
     .                       sho_file, clobber, append, status)
	      IF (status .NE. ferr_ok ) GOTO 2000
	      CALL SPLIT_LIST(0,  show_lun,
     .				cmnd_buff(arg_start(1):arg_end(1)), 0)
     
* ... close the file
	      CLOSE ( UNIT = show_lun, ERR = 2000 )
	      
* ... message logged in journal file
	   ELSEIF ( qual_given( slash_journal ) .GT. 0 ) THEN
	      IF (mode_journal .AND. jrnl_lun.NE.unspecified_int4) THEN
	         WRITE (jrnl_lun, '(A)')
     .				cmnd_buff(arg_start(1):arg_end(1))
	         RETURN
	      ELSE
	         RETURN   ! journal file isn't open
	      ENDIF

* ... message to operator
	   ELSE
	      CALL SPLIT_LIST(0,  ttout_lun,
     .				cmnd_buff(arg_start(1):arg_end(1)), 0)

	   ENDIF

* no message supplied. Just send a CR
        ELSEIF ( qual_given(slash_continue) .GT. 0) THEN
              WRITE ( ttout_lun, * ) 
        ENDIF

* continue or wait ?
	IF ( qual_given( slash_continue ) .GT. 0) RETURN

* do not wait if in server mode
        IF  (IS_SERVER()) then
           RETURN
        ENDIF

* pausing, so make sure the graphics (if any) are updated
        CALL FGD_CONSIDER_UPDATE(.TRUE.)

* prompt for <CR> ?
	IF ( qual_given( slash_quiet ) .EQ. 0 ) THEN
	   WRITE ( ttout_lun, * ) ' Hit Carriage Return to continue'
	ENDIF

* wait for continue signal (use TM_FRIENDLY_READ 12/92)
        CALL TM_FRIENDLY_READ(' ', risc_buff, *2000 )

 2000	RETURN
	END
