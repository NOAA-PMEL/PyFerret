	SUBROUTINE CHECK_GRAPHICS( status )

* check to see if there is a graphics output ready to recieve output
* (this routine assumes that PPLUS has been started)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  3/12/90 - written to permit metafile-only output

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include	'tmap_pplv11inc/gkscm1_inc.decl'	! with wsid
	include	'pplv11inc/GKSCM1.INC'	! with wsid
	include	'tmap_pplv11inc/pltcom_dat.decl'	! with gksopn
	include	'pplv11inc/PLTCOM.DAT'	! with gksopn
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE	'TMAP_PPLV11INC:GKSCM1_INC.DECL/LIST'	! with wsid
	INCLUDE	'PPLV11INC:GKSCM1.INC/LIST'	! with wsid
	INCLUDE	'TMAP_PPLV11INC:PLTCOM_DAT.DECL/LIST'	! with gksopn
	INCLUDE	'PPLV11INC:PLTCOM.DAT/LIST'	! with gksopn
#endif

* calling argument declarations
	INTEGER	status

* internal variable declarations
* ...

* only relevant for GKS output
	IF ( mode_gks ) THEN
	  IF ( wsid.EQ.0 .AND. .NOT.mode_metafile ) GOTO 5100
	ENDIF

* success
	status = ferr_ok
	RETURN

* not output device is ready
 5000	RETURN
 5100	CALL ERRMSG( ferr_state_not_set, status,
     .		     'no graphical output'//pCR//
     .		     'Use SET WINDOW or SET MODE METAFILE', *5000 )
	END
