      SUBROUTINE FERRET_PLOT_COMPLETE( n )

* Display a diagnostic message indicating that PLOT+ has completed its plot.
* This routine is to be called directly from PLOT+.
* It requires an External declaration in the main program to ensure
* that it is loaded at link time.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* Unix/RISC port - 10/91
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif
	
* calling argument declarations:
	INTEGER n

* local variable declarations:
	INTEGER slen
	CHARACTER LEFINT*5

        IF ( mode_diagnostic ) CALL SPLIT_LIST(pttmode_ops,
     .		ttout_lun, 'PPL plot '//LEFINT(n,slen)// 'complete', 0)

        RETURN
        END
