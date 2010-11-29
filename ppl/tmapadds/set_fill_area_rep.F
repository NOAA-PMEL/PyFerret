	SUBROUTINE SET_FILL_AREA_REP (zwsid,zwstype)

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
*	Setup the fill area indices for patterns or colors depending on
*	workstation characteristics

*	J Davison JISAO/PMEL/TMAP
*	9.2.88
*	JD modified 3.15.89 to support additional workstation types
*	2.18.92 *jd* mod to support ATC GKS

	integer	zwsid,zwstype

#ifdef unix
	include		'ws_types.cmn'
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
