	SUBROUTINE PROCESS_IF( cmnd_string, * )

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

* calling argument declarations:
	CHARACTER	cmnd_string*(*)
* RETURN 1 - used when the current command should be **skipped**

* internal variable declarations:
	INTEGER	STR_CASE_BLIND_COMPARE, vax_code, status	! not returned
	LOGICAL its_endif, its_else, its_elif, its_if

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/command.parm'
	include 'ferret_cmn/gfdl_vms.parm'
	include 'ferret_cmn/xprog_state.cmn'
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

* check for IF or ENDIF
	its_endif = STR_CASE_BLIND_COMPARE(cmnd_string(1:4),'ENDI')
     .		 .EQ. vms_str_success
	its_if   = (STR_CASE_BLIND_COMPARE(cmnd_string(1:3),'IF ')
     .		 .EQ. vms_str_success)
     .        .OR. (STR_CASE_BLIND_COMPARE(cmnd_string(1:3),'IF'//tab)
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
	   its_else  = STR_CASE_BLIND_COMPARE(cmnd_string(1:4),'ELSE')
     .	 	    .EQ. vms_str_success
	   its_elif  = STR_CASE_BLIND_COMPARE(cmnd_string(1:4),'ELIF')
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
