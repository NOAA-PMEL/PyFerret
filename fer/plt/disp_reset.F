	SUBROUTINE DISP_RESET

* initialize or reset the underlying plot package (PPLUS) to FERRET defaults
* (this routine is called only at start-up or by the command PPLUS/RESET)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/8/88 - extracted from DISP_INIT
* V200:   1/3/89 - added ALINE, AXLSZE
* V230:  11/4/92 - call SET_VIEWPORT(dflt_vp)
*        11/9/92 - unprotect all colors

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/plot_setup.parm'
	include 'ferret_cmn/xplot_setup.cmn'
	include 'ferret_cmn/xplot_state.cmn'
	include 'ferret_cmn/xrevision.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE 'FERRET_CMN:XPLOT_SETUP.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
	INCLUDE 'FERRET_CMN:XREVISION.CMN'
#endif

* local parameters
      LOGICAL     physical
      PARAMETER ( physical = .TRUE. )

* PLOT5 re-initialization (to forget previous states)
!	CALL PPLCMD ( from, line, 0, 'GET FERRET_PPLINIT', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'XFOR', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'YFOR', 1, 1 )
 	CALL PPLCMD ( from, line, 0, 'XLAB', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'YLAB', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'XAXIS', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'YAXIS', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'AXSET 1,1,1,1', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'PEN 0,1', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'PEN 1,1', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'ORIGIN 1.2,1.4', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'AXATIC 10,10', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'LABS', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'ALINE', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'TAXIS OFF', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'TXNMTC -1', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'TITLE', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'LABSET,0.16,0.12,0.12,0.12', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'WINDOW ON', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'BOX OFF', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'CROSS 0', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'AXLSZE .1,.1', 1, 1 )

	CALL SET_PLOT_SIZE( 10.2, 8.8 ) ! initial size
        CALL SET_VIEWPORT( mvp_dflt )   ! remove viewports, restore axis sizes

        CALL PPL_SHASET( 'RESET' )

	RETURN
	END
