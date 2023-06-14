	SUBROUTINE CLEANUP_LAST_CMND( control_stack_level )

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
* clean up the mess left by the previous command - esp. preparing a buffer
* of messages for the calling program, if any

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V312:  5/19/94 - with code extracted from GET_COMMAND
* V320:  11/94   - dont process error message buffer -- done in ferret_dispatch
* V531:  4/01 *sh* - cleanup incomplete continuation lines
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
* 10/2022 *acm* Code cleanup: remove references to obsolete gui interface

        IMPLICIT NONE
* calling argument declarations: (none)
	INTEGER	control_stack_level

* internal variable declarations:
	INTEGER status

	include 'tmap_dims.parm'
	include 'errmsg.parm'
	include 'ferret.parm'
	include 'command.parm'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xcontrol.cmn'
	include 'xinterrupt.cmn'

* re-store ^C interrupt handler following interrupt
 10	IF ( interrupted ) CALL ERRMSG
     .				( ferr_interrupt, status, ' ', *10 )
	ppl_interrupted = .FALSE.	! for interrupt of PPLUS

* send the state of the control stack back to the caller
* (zero is command line level)
	control_stack_level = csp

* this needed for logic in ferret_dispatch
	cmnd_buff = ' '

* this in case there was a half completed command entry
	has_continuation_cmnd = .FALSE.

	RETURN
	END
