	SUBROUTINE GET_VP_FROM_FERRET(  xorg, yorg,
     .					size, scale, xclip, yclip, xll, yll )

* get viewport and window information from FERRET common
* (this routine to hide FERRET common from PPLUS)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -  1/20/89
* V200:  3/13/90 - allow wsid.eq.0 for all windows cancelled (metafile open)
* V300:  3/20/93 - use ABS() to avoid negative size passed to PLOT+

#ifdef unix
	include 'ferret_cmn/ferret.parm'		! with unspecified_val4
	include 'ferret_cmn/xplot_state.cmn'		! viewport info
	include 'tmap_pplv11inc/gkscm1_inc.decl'	! with wsid
	include 'pplv11inc/GKSCM1.INC'	! with wsid
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'		! with unspecified_val4
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'		! viewport info
	INCLUDE 'TMAP_PPLV11INC:GKSCM1_INC.DECL'	! with WSID
	INCLUDE 'PPLV11INC:GKSCM1.INC'	! with WSID
#endif

* calling argument declarations:
	REAL xorg, yorg, size, scale, xclip, yclip, xll, yll

* internal variable declarations:
	LOGICAL clip

	xorg  = vp_xorg(vp_num)
	yorg  = vp_yorg(vp_num)
	size  = ABS(vp_size(vp_num))    ! 3/93 mod *sh*
	IF ( wsid .NE. 0 ) THEN
	   scale = wn_scale(wsid)
	ELSE
	   scale = 1.0
	ENDIF

	clip    = vp_xclip(vp_num) .NE. unspecified_val4
	IF ( clip ) THEN
	   xclip = vp_xclip( vp_num )
	   yclip = vp_yclip( vp_num )
	ENDIF

	xll = xlowlef(1)
	yll = ylowlef(1)

	RETURN
	END
