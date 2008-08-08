	SUBROUTINE XEQ_EXIT( * )

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
* execute the EXIT command
* if program is not under interactive control the /COMMAND_FILE qualifier is
* required

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/3/86
* revision 1.0 - 4/30/87 - EXIT in middle of command file will work
* revision 1.1 - 4/23/87 - flag arguments given as errors
* V312: 5/94 - use abnormal return for final program exit
* V314: 8/24/94 *kob* IBM port: slash_command_file should be declared as 
*				logical
* V510: *sh* 12/99 - close TMAP data sets when exiting
* V533: *kob* 8/01 - add checks for new EXIT qualifiers:
*	                 exit/loop
*			 exit/script
*                        exit/prompt
*			 exit/program
*                     With these new qualifiers, the routine has been
*                     reworked as well.
* V540: *sh* 11/01 - bug fix to logic in V5.33 change:
*	  "quit" in a command file was ignored
* V542: *kob* 11/02 - bug fix - check to make sure an exit w/ no qualifiers
*                               wasn't being called from a command file before
*                               completely exiting out of Ferret
* V581: *acm*  5/05 - fix bug 1233: exit/script needs to check whether there is
*                     an IF or REPEAT in the command stack
* V581: *acm*  5/05 - add EXIT/CYCLE to skip remaining commands in current 
*                     repetition of the loop
* V602: *acm*  3/07 - Fix bug 1304. If a script is executed with a loop, and
*                     exit/script is called, then exit just out of the script,
*                     not all the way back up out of the loop.
* V6.11 *acm*  4/08 - Fis bug 1566; EXIT/SCRIPT and IF blocks.
	
        include 'tmap_dims.parm'
	include 'command.parm'
	include 'errmsg.parm'
	include 'slash.parm'
	include 'xcontrol.cmn'
	include 'xprog_state.cmn'

	INTEGER status
	INTEGER i, pop, ifloop
	LOGICAL GO_FILE_INPUT

* make sure no arguments were given
	IF ( num_args .GT. 0 ) CALL ERRMSG( ferr_invalid_subcmnd, 
     .			status, cmnd_buff(arg_start(1):len_cmnd), *5000 )

	IF (qual_given(slash_exit_cycle) .GT. 0) THEN
* Skip to end of current repeat loop
	   pop=0
	   DO 100 i = csp,1,-1
	      IF ( cs_cmnd_num(i) .EQ. cmnd_repeat) GOTO 400
	      pop = pop + 1 
 100	   CONTINUE
	   RETURN        ! not in repeat loop, no need to move down stack

	ELSE IF (qual_given(slash_exit_loop) .GT. 0) THEN
* Exit from current repeat loop
	   pop=1
	   DO 200 i = csp,1,-1
	      IF ( cs_cmnd_num(i) .EQ. cmnd_repeat) GOTO 400
	      pop = pop + 1 
 200	   CONTINUE
	   RETURN        ! not in repeat loop, no need to move down stack
	ELSE IF (qual_given(slash_exit_script) .GT. 0) THEN

* Exit from current script.  Need to do check to make sure this
* wasn't executed from waiting Ferret prompt, ie not in a script
* Fix bug 1304 here; count pop up to the *GO current script*
* Fix bug 1566 here; multi-linen IFs are not listed in cs_cmnd_num()
* instead we need to check if_cs

	   IF (csp .GT. 0) THEN
              pop = 1
              DO 300 i = csp, 1, -1
                 IF (cs_cmnd_num(i) .EQ. cmnd_repeat) pop = pop + 1
                 IF (cs_cmnd_num(i) .EQ. cmnd_if) pop = pop + 1
                 
! Pop out of possibly nested if in current script (fixes bug 1566)
                 IF (ifstk .GT. 0) THEN 
                 IF (if_cs(ifstk) .EQ. i) THEN 
                    DO WHILE (if_cs(ifstk) .EQ. i .AND. ifstk.GT.1)
                       ifstk = ifstk - 1
                    ENDDO
!!!                    ifstk = ifstk + 1  ! ifstk will be popped once by POP_CS 
                 ENDIF
                 ENDIF

                 IF (cs_cmnd_num(i) .EQ. cmnd_semicolon) pop = pop + 1
                 IF (cs_cmnd_num(i) .EQ. cmnd_go) GOTO 400  ! Reached the next GO level(fixes bug 1566)
 300	      CONTINUE
              GOTO 400
	   ENDIF
	ELSE IF (qual_given(slash_exit_prompt) .GT. 0) THEN
* Exit to Ferret prompt
	   pop = csp
	   GOTO 400
	ELSE IF (qual_given(slash_exit_program) .GT. 0    .OR.
     .  	  (qual_given(slash_exit_command) .GT. 0) .OR. 
     .            .NOT. GO_FILE_INPUT()) THEN
* Exit program completely 
* *kob* new call to GO_FILE_INPUT to make sure were not in command file
	   CALL CLSPPL		! close PLOT5
	   CALL TM_EXIT		! close TMAP data sets
	   RETURN 1
!	   IF ( frame_on ) CALL MOVIE_CLOSE( frame_lun, status )! end movie
	ELSE
* no qualifier supplied on a "quit" in a command file: Exit current script
	   CALL POP_CS
	   RETURN
	ENDIF

* Here is where we'll pop the stack once for each level we need to move down
 400	IF (pop .GT. 0) then
	   DO 500 i = 1,pop
	      CALL POP_CS
 500	   CONTINUE
	ENDIF
* error exit
 5000	RETURN

	END





