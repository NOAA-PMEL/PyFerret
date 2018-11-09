	SUBROUTINE TPLOT_AXIS_ENDS (plt_ax_lo, plt_ax_hi, cal_id, tstyle)

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
* 11/2016 ACM, see ticket 2483

* Reset plt_ax_lo, plt_ax_hi to the values that PPLUS will use for the
* ends of the time axis that it draws. plt_ax_lo are in seconds since 0000
* This must be preceded by a call to TAXIS_STYLE to set t_axtyp.


        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'xplot_setup.cmn'

	REAL plt_ax_lo, plt_ax_hi
	INTEGER cal_id
	CHARACTER*3 tstyle

	INTEGER t_axtyp
	REAL	WHOI2BC
	CHARACTER*14 ABS_WHOI_DATE, whoimin, whoimax, 
     .			whoimin_axlo, whoimax_axhi


* only need this call once for a line plot.

	IF (adjust_time) RETURN

* translate to WHOI-formatted dates
	whoimin =  ABS_WHOI_DATE( plt_ax_lo, cal_id )
	whoimax =  ABS_WHOI_DATE( plt_ax_hi, cal_id )

* put the century digits at the end, as FIXTIM expects
	whoimin = whoimin(3:14)//whoimin(1:2)
	whoimax = whoimax(3:14)//whoimax(1:2)

* t_axtyp is the time style used by PPLUS

	if (tstyle.EQ.'MIN') then
	   t_axtyp=0
	else if (tstyle(1:1).EQ.'H') then
	   t_axtyp=0
	else if (tstyle(1:1).EQ.'D') then
	   t_axtyp=0
	else if (tstyle(1:1).EQ.'M') then
	   t_axtyp=1
	else
	   t_axtyp=3
	endif

	CALL FIXTIM (whoimin, whoimax, whoimin_axlo, whoimax_axhi, t_axtyp)

* Translate back to seconds-since 0000
	plt_ax_lo = 60.* WHOI2BC( whoimin_axlo )
	plt_ax_hi = 60.* WHOI2BC( whoimax_axhi ) 

	adjust_time = .TRUE.

	RETURN
	END
