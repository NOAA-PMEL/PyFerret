	SUBROUTINE SET_FILL_AREA_REP (zwsid,zwstype)

*	Setup the fill area indices for patterns or colors depending on
*	workstation characteristics

*	J Davison JISAO/PMEL/TMAP
*	9.2.88
*	JD modified 3.15.89 to support additional workstation types
*	2.18.92 *jd* mod to support ATC GKS

	integer	zwsid,zwstype

#ifdef unix
	include		'tmap_pplv11inc/ws_types.cmn'
#else
	include		'tmap_pplv11inc:ws_types.inc'
#endif

******************************************************************************
#ifdef atc_gks
	if (zwstype/100 .eq. ws_xwindow/100 .or. zwstype .eq. ws_uis) then
#else
	if (zwstype .eq. ws_xwindow .or. zwstype .eq. ws_uis) then
#endif
*	   XWINDOW (OR UIS) WS
	   call ws_fill_rep (zwsid,zwstype)

	else if (zwstype .eq. ws_ps_mono .or. zwstype .eq. ws_tek4014) then
*	   MONOCHROME HATCHING
	   call hatch_fill_rep (zwsid)

	else if (zwstype .eq. ws_tek4107) then
*	   TEK 4107
	   call tek4107_fill_rep (zwsid)

	else
*	   SETUP DEFAULT FILL AREA BUNDLES
	   call ws_fill_rep (zwsid,zwstype)
	end if

	return

	end
