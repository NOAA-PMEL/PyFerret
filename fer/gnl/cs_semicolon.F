	SUBROUTINE CS_SEMICOLON( cmnd_line, * )

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
* this routine is called to save a group of commands separated by semicolons
* and bump the FERRET "control stack" to place that group of commands in
* control

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V3.00 - 2/1/93

* calling argument declarations:
	CHARACTER*(*)	cmnd_line
* RETURN 1 - always taken

* internal variable declarations:
	INTEGER	TM_LENSTR1, status

#ifdef unix
	include 'errmsg.parm'
	include 'ferret.parm'
	include 'command.parm'
	include 'xprog_state.cmn'
	include 'xcontrol.cmn'
#else
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:COMMAND.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XCONTROL.CMN'
#endif

* allocate next spot on the control stack
* ... but leave the stack in place in case of errors in this routine
	CALL STACK_PTR_UP( csp, cs_max, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* load the command group
	cs_text (csp) = cmnd_line
        cs_phase(csp) = 0    ! points before the first character
        cs_limit(csp) = TM_LENSTR1( cmnd_line )

* and tell 'em who owns this level
	cs_cmnd_num( csp ) = cmnd_semicolon

* tell 'em we've taken over and start workin'
	cs_in_control = .TRUE.
	input_source  = input_from_cs
	RETURN 1

* error exit
* ... control stack pointer, CSP, corrected in ERRMSG
 5000	RETURN
	END
