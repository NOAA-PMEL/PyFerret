      CHARACTER*(*) FUNCTION GKS_X_CONID( wsid )

* determine the correct name for the CONID to open the next GKS output window
* the value of wsid is assumed to be 1 for the first window, 2,3,... for
* subsequent windows
* the name stored in xwind_node is assumed to end with ":0" or "::0"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200: 12/20/89
* RISC/Ultrix port *sh* 5/91:  allow tcp/ip addresses, too 
* V230:  5/14/92 - REMOTE_X mode was replaced by the REFRESH mode

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif
	
* calling argument declarations:
	INTEGER		wsid

* internal variable declarations:
	CHARACTER	letter*1

* generate letter "A", "B", ...
	letter = CHAR(64+wsid)

* format of conid is DECnet_node::0.letter
*                 or tcp.ip.node:0.letter
* the stored name (xwind_node) lacks only the "dot-letter" ending
* eliminated mode_Xwindows 5/92 
!	IF ( mode_Xwindows ) THEN
!           GKS_X_CONID = xwind_node(:xwind_node_len)//'.'//letter
!	ELSE
* ... local DECnet connect name
	  GKS_X_CONID = '0::0.'//letter
!	ENDIF

	RETURN
	END
