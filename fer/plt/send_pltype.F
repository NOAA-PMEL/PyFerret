	SUBROUTINE SEND_PLTYPE( meta )

* send the PLTYPE command to PPLUS including workstation type and metafile
* information

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:   1/5/90
* V230:  2/11/92 - (SUN port) fixed bug that "WS=DEFAULT" was sent to PPLUS
*        11/9/92 - eliminate device dependence (no WS=, no META=)
* V310:  12/4/94 - use mode_GKS to determine what PLTYPE to send

* calling argument declarations:
	LOGICAL meta

* local veriable declarations:
	INTEGER	TM_LENSTR1

#ifdef unix
	include 'ferret_cmn/plot_setup.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xplot_setup.cmn'
	include 'ferret_cmn/xgks_devices.cmn'
#else
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_SETUP.CMN'
	INCLUDE 'FERRET_CMN:XGKS_DEVICES.CMN'
#endif

* command format: "PLTYPE 3 META"
	IF ( mode_gks ) THEN
	   ppl_buff = 'PLTYPE 3'
	ELSE
	   ppl_buff = 'PLTYPE 1'	!  4/94
	ENDIF

* append META=device
	IF ( meta ) ppl_buff = ppl_buff(:8)//' META'

* send it
	CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )

	RETURN
	END
