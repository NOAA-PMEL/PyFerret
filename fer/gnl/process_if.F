	SUBROUTINE PROCESS_IF( cmnd_string, slen, * )

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
* this routine is called when the process of obtaining Ferret commands is
* under the control of a multi-line IF clause
* This routine inspects the currently available command and decides whether
* to execute it or to skip it. It also updates the IF-stack as needed.

* Note that error checking need not be done in this routine because the
* "logical control" tokens "IF", "ELSE", "ELIF", and "ENDIF" are themselves
* commands which will check for errors and set appropriate states

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420:  4/28/96
*     : bug fix 5/96: ENDIF processed wrong for nested IF
* V530: 1/01 *sh* - skip white space at start of line

* calling argument declarations:
	INTEGER		slen
	CHARACTER	cmnd_string*(*)
* RETURN 1 - used when the current command should be **skipped**

* internal variable declarations:
	INTEGER	STR_CASE_BLIND_COMPARE, ic1,
     .		status	! not returned
	LOGICAL its_endif, its_else, its_elif, its_if

#ifdef unix
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'command.parm'
	include 'gfdl_vms.parm'
	include 'xprog_state.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ESSMGS.PARM'
	INCLUDE 'FERRET_CMN:COMMAND.PARM'
	INCLUDE 'FERRET_CMN:GFDL_VMS.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* internal parameter declarations:
        CHARACTER*1     tab
        PARAMETER     ( tab = CHAR(9) )

* sanity check
	IF ( ifstk.LT.1 .OR. ifstk.GT.max_ifstk ) GOTO 5000

* skip initial white space
	DO ic1 = 1, slen
	   IF (cmnd_string(ic1:ic1).NE. ' '
     .   .AND. cmnd_string(ic1:ic1).NE. tab) GOTO 100
	END DO
! should never arrive here ...
	ic1 = 1

* check for IF or ENDIF
 100	its_endif =
     .	      STR_CASE_BLIND_COMPARE(cmnd_string(ic1:ic1+3),'ENDI')
     .		 .EQ. vms_str_success
	its_if   =
     .	     (STR_CASE_BLIND_COMPARE(cmnd_string(ic1:ic1+2),'IF ')
     .		 .EQ. vms_str_success)
     .  .OR. (STR_CASE_BLIND_COMPARE(cmnd_string(ic1:ic1+2),'IF'//tab)
     .		 .EQ. vms_str_success)

* always execute the IF and ENDIF statements
	IF (its_if .OR. its_endif) RETURN

* now check each of the possible IF execution states
	IF ( if_doing(ifstk) .EQ. pif_skip_to_endif ) THEN
* skip everything until ENDIF is encountered
	   RETURN 1

	ELSEIF ( if_doing(ifstk) .EQ. pif_doing_clause ) THEN
* execute everything cuz we're "doing" the clause
	   RETURN

	ELSEIF ( if_doing(ifstk) .EQ. pif_skip_to_clause ) THEN
* check for ELSE or ELIF
	   its_else  =
     .	      STR_CASE_BLIND_COMPARE(cmnd_string(ic1:ic1+3),'ELSE')
     .	 	    .EQ. vms_str_success
	   its_elif  =
     .	      STR_CASE_BLIND_COMPARE(cmnd_string(ic1:ic1+3),'ELIF')
     .		    .EQ. vms_str_success
	   IF (its_else .OR. its_elif) RETURN
* nope - just an ordinary command ... keep skipping 
	   RETURN 1

	ELSE
* unknown if condition
	   GOTO 5000

	ENDIF
 1000	RETURN


 5000	CALL ERRMSG(ferr_internal, status, 'ifstk!!', *1000)
	END
