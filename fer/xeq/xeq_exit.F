	SUBROUTINE XEQ_EXIT( * )

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

#ifdef unix
	include 'tmap_dims.parm'
	include 'command.parm'
	include 'errmsg.parm'
	include 'slash.parm'
	include 'xcontrol.cmn'
	include 'xprog_state.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

	INTEGER status
	LOGICAL slash_command_file

	INTEGER i, pop

* make sure no arguments were given
	IF ( num_args .GT. 0 ) CALL ERRMSG( ferr_invalid_subcmnd, 
     .			status, cmnd_buff(arg_start(1):len_cmnd), *5000 )


	IF (qual_given(slash_exit_loop) .GT. 0) THEN
* Exit from current repeat loop
	   pop=1
	   DO 100 i = csp,1,-1
	      IF ( cs_cmnd_num(i) .EQ. cmnd_repeat) GOTO 400
	      pop = pop + 1 
 100	   CONTINUE
	   RETURN        ! not in repeat loop, no need to move down stack
	ELSE IF (qual_given(slash_exit_script) .GT. 0) THEN
* Exit from current script.  Need to do check to make sure this
* wasn't executed from waiting Ferret prompt, ie not in a script
	   IF (csp .GT. 0) CALL POP_CS
	   RETURN
	ELSE IF (qual_given(slash_exit_prompt) .GT. 0) THEN
* Exit to Ferret prompt
	   pop = csp
	   goto 400
	ELSE IF (qual_given(slash_exit_program) .GT. 0    .OR.
     .  	  (qual_given(slash_exit_command) .GT. 0) .OR.
     .            input_source .NE. input_from_file) THEN
* Exit program completely
	   CALL CLSPPL		! close PLOT5
	   CALL TM_EXIT		! close TMAP data sets
	   RETURN 1
!	   IF ( frame_on ) CALL MOVIE_CLOSE( frame_lun, status )! end movie
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

