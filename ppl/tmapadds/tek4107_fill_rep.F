	SUBROUTINE TEK4107_FILL_REP (zwsid)

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
*	Setup the fill area indices for colors on TEK 4107 WS

*	J Davison JISAO/PMEL/TMAP
*	9.2.88/3.15.89
*	*jd* 2.19.92 mod to support atc gks

#ifdef unix
        include         'shade_vars.cmn'
	include		'ws_types.cmn'
#else
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
	include		'tmap_pplv11inc:ws_types.inc'
#endif

#ifdef atc_gks
#   ifdef unix
	include        'gkspar.inc'
#   else
	INCLUDE        'atc_gksdir:gkspar.inc'
#   endif
#else
#   ifdef unix
  	include        'gkspar.inc'
#   else
  	INCLUDE        'sys$library:gksdefs.bnd'
#   endif
#endif

	integer		zwsid,color,ndx

********************************************************************************

*	SET THE COLOR TABLE TO CUSTOM COLORS OR TO THE DEFAULT SELECTION
	if (custom_colors) then 
	   call setup_color_table (zwsid,2,14)
	else	
	   call set_default_colors (zwsid,ws_tek4107,2,14)
	end if

*	GO THRU AVAILABLE INDICES FOR FILL REP
	do 100 ndx = 1,shd_levels
	   color = 2 + (ndx-1)*14/(shd_levels)
	   call gsfar (zwsid,ndx+1,gsolid,1,color)
100	continue
	return

	end
