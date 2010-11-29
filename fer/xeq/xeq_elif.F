	SUBROUTINE XEQ_ELIF

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
* Process the intermediate logic of an ELIF ("ELSEIF") statement in a
* multi-line IF compound statement

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V420:  4/96

#ifdef unix
	include	'ferret.parm'
	include	'command.parm'
	include 'errmsg.parm'
	include 'gfdl_vms.parm'
	include 'xcontext.cmn'
	include 'xcontrol.cmn'
	include 'xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:COMMAND.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:GFDL_VMS.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XCONTROL.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* local variable declarations
	LOGICAL TRUE_OR_FALSE, condition
	INTEGER	STR_CASE_BLIND_COMPARE,
     .		vax_code, status

* initial sanity checks
	IF ( .NOT.if_conditional ) GOTO 5010

* is this the end of the clause that has been executing?
	IF ( if_doing(ifstk) .EQ. pif_doing_clause ) THEN
	   if_doing(ifstk) = pif_skip_to_endif
	   RETURN
	ENDIF

* more sanity checks
* (not the best error message ... but this is an internal never-happen error
* since pif_skip_to_endif is supposed to be trapped outside of this)
	IF ( if_doing(ifstk) .NE. pif_skip_to_clause ) GOTO 5010

* check that something was given to "if" on
	IF ( num_args .LT. 2 ) GOTO 5100

* optional "THEN" at end of line is OK
	IF ( num_args .GT. 1 ) THEN
	   IF ( num_args .GT. 2 ) GOTO 5300
	   vax_code = STR_CASE_BLIND_COMPARE(
     .				cmnd_buff(arg_start(2):arg_end(2)),
     .				'THEN' )
	   IF (vax_code .NE. vms_str_success) GOTO 5300
	ENDIF

* determine if the condition is true or false
	condition = TRUE_OR_FALSE(cmnd_buff(arg_start(1):arg_end(1)),
     .				  status)
	IF ( status .NE. ferr_ok ) GOTO 5000

* say which IF clause to execute
	IF ( condition ) THEN
	   if_doing(ifstk) = pif_doing_clause	! ELIF TRUE THEN (do it)
	ELSE
	   if_doing(ifstk) = pif_skip_to_clause	! ELIF FALSE THE (keep looking)
	ENDIF

	RETURN
	
* error exit
* ... control stack pointer, CSP, corrected in GFD_ERROR_HANDLER
 5000	RETURN
 5010	CALL ERRMSG( ferr_invalid_command, status,
     .		'ELIF can only be used between IF and ENDIF', *5000)
 5100	CALL ERRMSG( ferr_syntax, status, 'ELIF what ?', *5000)
 5300	CALL ERRMSG( ferr_syntax, status,
     .		'Entire ELIF statement should be "ELIF expr THEN"'//
     .			pCR//cmnd_buff(:len_cmnd), *5000 )
	END
