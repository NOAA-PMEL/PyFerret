	SUBROUTINE GET_VP_FROM_FERRET(  xorg, yorg,
     .					size, scale, xclip, yclip, xll, yll )

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
* get viewport and window information from FERRET common
* (this routine to hide FERRET common from PPLUS)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -  1/20/89
* V200:  3/13/90 - allow wsid.eq.0 for all windows cancelled (metafile open)
* V300:  3/20/93 - use ABS() to avoid negative size passed to PLOT+
* V540:  *sh* 9/01 - call VP_AXLIM2LIM to translate axis-relative VP limits

	include 'ferret.parm'		! with unspecified_val4
	include 'xplot_state.cmn'		! viewport info
	include 'gkscm1_inc.decl'	! with wsid
	include 'GKSCM1.INC'	! with wsid

* calling argument declarations:
	REAL xorg, yorg, size, scale, xclip, yclip, xll, yll

* internal variable declarations:
	LOGICAL clip
	REAL  xoei,  yoei,  xcei,  ycei

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

* 9/01 additions
	IF (vp_by_axis( vp_num )) THEN
* ... adjust limits given as axis positions to limits of viewport
	   CALL VP_AXLIM2LIM( scale, xorg, yorg, xclip, yclip,
     .                               xorg, yorg, xclip, yclip,
     .                               xoei,  yoei,  xcei,  ycei )
	ENDIF

	RETURN
	END
