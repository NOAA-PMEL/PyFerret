	SUBROUTINE SETUP_LINE_BUNDLES (zwsid,zwstype)

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
*	Routine to set polyline indices bundled and set up the bundles
*	for WS zwsid of type zwstype.
*	J Davison JISAO/PMEL/TMAP 3.23.89
*	*jd* 2.19.92 mod to support atc gks

#ifdef unix
	include		'ws_types.cmn'
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

