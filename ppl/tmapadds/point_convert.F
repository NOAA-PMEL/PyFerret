	SUBROUTINE POINT_CONVERT (xno_uzr,yno_uzr,xuzr,yuzr)

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
*	Converts a point (x,y) in /nouser coordinates to /user
*	J Davison 10.17.89

#ifdef unix
	include		'tmap_pplv11inc/plt_inc.decl'
	include		'pplv11inc/PLT.INC'
	include		'tmap_pplv11inc/axis_inc.decl'
	include		'pplv11inc/AXIS.INC'
#else
	INCLUDE		'tmap_pplv11inc:tmap_plt.inc'
	INCLUDE		'tmap_pplv11inc:tmap_axis.inc'
#endif

	real		xno_uzr,yno_uzr,xuzr,yuzr,xf,yf

********************************************************************************

	xf = (xhi - xlo)/xlen
	yf = (yhi - ylo)/ylen

	xuzr = xno_uzr*xf + xlo
	yuzr = yno_uzr*yf + ylo
                                                                               
	if (itypex .gt. 1) xuzr = 10.0**xuzr
	if (itypey .gt. 1) yuzr = 10.0**yuzr

	return
	end
