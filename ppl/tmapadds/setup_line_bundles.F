	SUBROUTINE SETUP_LINE_BUNDLES (zwsid,zwstype)

*	Routine to set polyline indices bundled and set up the bundles
*	for WS zwsid of type zwstype.
*	J Davison JISAO/PMEL/TMAP 3.23.89
*	*jd* 2.19.92 mod to support atc gks

#ifdef unix
	include		'tmap_pplv11inc/ws_types.cmn'
#else
	include		'tmap_pplv11inc:ws_types.inc'
#endif

	integer		zwsid,zwstype

******************************************************************************

*	SET UP LINE BUNDLE FOR 'PEN N,0' CALLS -- USE INDEX 19 JD 2.19.92
        call gsplr (zwsid,19,1,1.0,0)

#ifdef atc_gks
	if (zwstype/100 .eq. ws_xwindow/100 .or. zwstype .eq. ws_uis) then
#else
	if (zwstype .eq. ws_xwindow .or. zwstype .eq. ws_uis) then
#endif
*	   XWINDOW (OR UIS) WORKSTATION 
	   call ws_line_bundle (zwsid,zwstype)

	else if (zwstype .eq. ws_ps_mono) then
*	   MONOCHROME POSTSCRIPT ATTRIBUTES
	   call ps_line_bundle (zwsid)

	else if (zwstype .eq. ws_tek4014) then
*	   TEK 4014
	   call tek4014_line_bundle (zwsid)

	else if (zwstype .eq. ws_tek4107) then
*	   TEK 4107
	   call tek4107_line_bundle (zwsid)

	else
*	   SETUP DEFAULT LINE BUNDLES
	   call ws_line_bundle (zwsid,zwstype)
	end if



	return

	end

