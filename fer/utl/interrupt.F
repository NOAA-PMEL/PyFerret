	SUBROUTINE INTERRUPT ( * )

* send appropriate signals to indicate an interrupt has taken place
* always take "abnormal" return

* ... FERRET version ( different version for PPLUS )

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 1/24/89

#ifdef unix
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xplot_state.cmn'
#else
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
#endif

* local variable declarations:
	INTEGER	status

* set flag indicating that interrupt was detected and resolved in PPLUS
	ppl_interrupted = .TRUE.

* issue interrupted message and set FERRET states to respond
	CALL ERRMSG ( ferr_interrupt, status, ' ', *5000 )

 5000	RETURN 1
	END
