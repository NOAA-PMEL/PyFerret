      SUBROUTINE CK_GKS_ACTIVE( * )

* check if GKS is avtive - if not declare error and take abnormal return

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 12/20/89

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'tmap_pplv11inc/pltcom_dat.decl'   ! with gksopn
	include 'pplv11inc/PLTCOM.DAT'   ! with gksopn
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'TMAP_PPLV11INC:PLTCOM_DAT.DECL/LIST'   ! with gksopn
	INCLUDE 'PPLV11INC:PLTCOM.DAT/LIST'   ! with gksopn
#endif

	external errmsg

* calling argument declarations:
*	...

* internal variable declarations:
	INTEGER	state, status

* irrelevant for PLTYPE .NE. 3 or 4
	IF ( .NOT.gksopn ) RETURN

* get GKS operating state
	CALL GQOPS( state )

* states:
*	0 - GKS closed
*	1 - GKS open
*	2 - At least 1 workstation is open
*	3 - At least 1 workstation is active
*	4 - a segment is open

	IF ( state .GE. 3 ) RETURN
	   
* GKS isn't ready !!
	CALL ERRMSG( ferr_state_not_set, status,
     .		'graphical output device isnt ready', *5000 )
 5000	RETURN 1

	END
