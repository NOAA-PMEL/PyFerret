	SUBROUTINE MOUSE_COMMAND

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
* get next command line from the GUI input source
* if it is a coded query respond instantly

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V300:  6/16/93
* V301:  1/94 - bug:  incorrect call to DO_QUERY in error return
* V314: 8/19/94 *kob* IBM port - Modified format statements, added 
*				 ifdef MANDATORY_FORMAT_WIDTHS for I format
*			         descriptor 
* *kob* 1/03 - replace call to intrinsic func CHAR with its octal
*              equivalent - needed for g77 port

#ifdef unix
	include 'errmsg.parm'
	include 'ferret.parm'
	include 'command.parm'
	include 'xprog_state.cmn'
	include 'xinterrupt.cmn'
	include 'xgui.cmn'
#else
	INTEGER	SMG$READ_COMPOSED_LINE, vax_code

	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:COMMAND.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XINTERRUPT.CMN'
	INCLUDE 'FERRET_CMN:XGUI.CMN'
#endif

* calling argument declarations: (none)

* internal variable declarations:
	LOGICAL previous_eof, reverify, looking_for_end
	INTEGER	TM_LENSTR, TM_LENSTR1,
     .		status, iqual, ilist, cmnd_len, query, i

        CHARACTER*1 backslash
        PARAMETER ( backslash = O'134' )


* read from standard input without prompt
 10	READ ( ttin_lun, '(A)' ) cmnd_buff

* Check for for QUERY commands beginning with "gui_char"
* These commands are handled by this routine without returning to FERRET
	IF ( cmnd_buff(1:1) .EQ. gui_char ) THEN
	   IF ( cmnd_buff(2:2) .EQ. '>' ) THEN
* ">" is a "CONTINUE" signal from GUI  (not used by GUI ??)
	      cmnd_buff = ' '
	      RETURN
	   ELSE
* parse the query: format of command is blank-only separated:
*	"^nn tag arg2 arg3" 
*      (argument numbering to match "QUERY ARG1 arg2 arg3 ...")
* the "tag" field is not a standard part of the QUERY command
* it is a special field which is simply echoed by FERRET to help the 
* GUI keep track of replies
#ifdef MANDATORY_FORMAT_WIDTHS
* no fixed field width works but "*" may cause problems too
* such as "Y" being an unacceptable integer
	      READ ( cmnd_buff(2:3), *, ERR=5000 ) query
#else
	      READ ( cmnd_buff(2:3), '(I)', ERR=5000 ) query
#endif
	      len_cmnd = TM_LENSTR1( cmnd_buff )
	      num_args = 0
	      looking_for_end = .TRUE.
	      DO 100 i = 2, len_cmnd
* find end of this argument (starting with pseudo-argument 0 which is "^nn")
	         IF ( looking_for_end ) THEN
		   IF ( cmnd_buff(i:i) .EQ. " " ) THEN
* ... record end of arg
	              IF ( num_args .GE. 1 ) arg_end(num_args) = i-1
	              looking_for_end = .FALSE.
	           ENDIF
	         ELSE
* find start of arg
	            IF ( cmnd_buff(i:i) .NE. " " ) THEN
* ... record start of next arg
	               num_args = num_args + 1
	               arg_start(num_args) = i
	               looking_for_end = .TRUE.
	            ENDIF
	         ENDIF
 100	      CONTINUE
	      IF ( num_args .GE. 1 ) arg_end(num_args) = len_cmnd

* answer query and request next command from GUI without ever bothering FERRET
	      CALL DO_QUERY( err_lun, gui_fmt1, query,
     .	                     cmnd_buff(arg_start(1):arg_end(1)), status )
	      GOTO 10
	   ENDIF
	ELSE

* normal command - pass it along to FERRET
	   RETURN
	ENDIF

* synax error in GUI query - give a no-content response
 5000	CALL DO_QUERY( err_lun, gui_fmt1, 9999, '*', status )
	GOTO 10
	END
