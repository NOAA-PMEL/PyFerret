	SUBROUTINE SET_AX_SIZES( xfrac, yfrac )

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
* adjust PPLUS AXLEN and ORIGIN commands to fit current plot size
* xfrac and yfrac allow viewports to specify fractions of the window size

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V230:  11/5/92 - removed from SET_PLOT_SIZE for auto-viewports
* V301:  11/3/93 - set PLOT+ symbols VP_WIDTH and VP_HEIGHT

#ifdef unix
	include 'ferret_cmn/plot_setup.parm'
        include 'ferret_cmn/xplot_setup.cmn'      ! 10/91 for saved_ppl_yax
        include 'tmap_pplv11inc/plt_inc.decl'     ! with width/height
        include 'pplv11inc/PLT.INC'
#else
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
        INCLUDE 'FERRET_CMN:XPLOT_SETUP.CMN'      ! 10/91 for saved_ppl_yax
        INCLUDE 'TMAP_PPLV11INC:PLT_INC.DECL'     ! with width/height
        INCLUDE 'PPLV11INC:PLT.INC'
#endif

* calling argument declarations:
        REAL xfrac, yfrac

* internal variable declarations:
	INTEGER slen
	REAL	xaxlen, yaxlen
	REAL	xsize, ysize
	CHARACTER TM_FMT*8, buff*24

* calculate plot size
	xsize = xfrac*width
	ysize = yfrac*height
        xaxlen = MAX( 0.001, xsize - 2.2 )
        yaxlen = MAX( 0.001, ysize - 2.8 )

* tell axis lengths to PPLUS
	WRITE ( buff, 3000 ) xaxlen, yaxlen
	CALL PPLCMD ( from, line, 0, 'AXLEN '//buff, 1, 1 )
 3000	FORMAT(2F12.3)

* define viewport size symbols to PLOT+ (11/93)
	xsize  = MAX( 0.001, xsize )
	ysize  = MAX( 0.001, ysize )
	buff = TM_FMT(xsize,-3,8,slen) 
	CALL PPLCMD ( from, line, 0, 'SET VP_WIDTH '//buff, 1, 1 )
	buff = TM_FMT(ysize,-3,8,slen) 
	CALL PPLCMD ( from, line, 0, 'SET VP_HEIGHT '//buff, 1, 1 )

* unsave previous values and restore the origin
        saved_ppl_yax = .FALSE.
        CALL PPLCMD ( from, line, 0, 'ORIGIN 1.2,1.4', 1, 1 )

	RETURN
	END
