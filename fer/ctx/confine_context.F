	SUBROUTINE CONFINE_CONTEXT( cx, status )

* given a skeletal context with a known grid confine the context limits to
* within the grid limits
* if date/time limits were specified convert to time steps here

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 5/5/88  - code lifted from FINISH_1_CONTEXT_MOD
* revision 0.1 - 7/8/88  - avoid wiping out half of xy_ave transform
* revision 0.2 - 7/28/88 - simplified date processing (no mixed dates/tsteps)
* revision 0.3 - 1/12/89 - new error report for munknown axis
* V200:  5/26/89 - removed bulk of routine to CONFINE_AXIS
* V230:  7/20/92 - bug in error trapping fixed
* *kob* 11/96   - Linux port.  Linux didn't like tabs between "include"
*                 and what was being included.  removed them
#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations:
	INTEGER	cx, status

* internal variable declarations:
	INTEGER	idim

	DO 500 idim = 1, 4
    	   CALL CONFINE_AXIS( idim, cx, status )
           IF ( status .NE. ferr_ok ) RETURN      ! 7/92 bug fix
 500    CONTINUE

	RETURN
	END
