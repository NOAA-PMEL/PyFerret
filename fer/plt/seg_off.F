      SUBROUTINE SEG_OFF()

* turn off current GKS segment

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 6/30/88

* calling argument declarations:
* ...

* local variable declarations:
* ...

#ifdef unix
	include	'ferret_cmn/xprog_state.cmn'		! with mode_segment
	include 'tmap_pplv11inc/pltcom_dat.decl'	! with gksopn
	include 'pplv11inc/PLTCOM.DAT'	! with gksopn
#else
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'		! with mode_segment
	INCLUDE 'TMAP_PPLV11INC:PLTCOM_DAT.DECL/LIST'	! with gksopn
	INCLUDE 'PPLV11INC:PLTCOM.DAT/LIST'	! with gksopn
#endif


	IF ( .NOT. mode_segment			! not using segments
     .	.OR. .NOT. gksopn	) RETURN	! not a VAXstation

* close current segment
	CALL GCLSG( )

	RETURN
	END
