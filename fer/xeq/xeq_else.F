	SUBROUTINE XEQ_ELSE

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

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V420:  4/96

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* local variable declarations
	INTEGER	status

* initial sanity checks
	IF ( .NOT.if_conditional ) GOTO 5010

* check that something was given to "if" on
	IF ( num_args .GE. 1 ) GOTO 5100

* is this the end of the clause that has been executing?
	IF ( if_doing(ifstk) .EQ. pif_doing_clause ) THEN
	   if_doing(ifstk) = pif_skip_to_endif

* or the start of the clause to execute
	ELSEIF ( if_doing(ifstk) .EQ. pif_skip_to_clause ) THEN
	   if_doing(ifstk) = pif_doing_clause

* anything else is an internal error
	ELSE
	   GOTO 5200
	ENDIF

	RETURN
	
* error exit
* ... control stack pointer, CSP, corrected in GFD_ERROR_HANDLER
 5000	RETURN
 5010	CALL ERRMSG( ferr_invalid_command, status,
     .		'ELSE can only be used between IF and ENDIF', *5000)
 5100	CALL ERRMSG( ferr_syntax, status,
     .		'ELSE statement stands alone. Continue on next line"'//
     .			pCR//cmnd_buff(:len_cmnd), *5000 )
 5200	CALL ERRMSG( ferr_internal, status, 'IF-ELSE', *5000)
	END
