      SUBROUTINE VP_AXLIM2LIM( scale, xloax, yloax, xhiax, yhiax,
     .                                xlovp, ylovp, xhivp, yhivp,
     .                                xoei,  yoei,  xcei,  ycei )

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
* Determine the viewport limits that are implied from viewport limits
* specified ax axis endpoint positions

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V540: 9/01 *sh*

	include 'ferret.parm'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xplot_setup.cmn'
        include 'plt_inc.decl'     ! with width/height
        include 'PLT.INC'
	
* calling argument declarations:
	REAL    scale, xloax, yloax, xhiax, yhiax,
     .                 xlovp, ylovp, xhivp, yhivp,
     .                 xcei, ycei, xoei, yoei

* internal variable declarations:
	REAL	xcv, xov, ycv, yov, xwpi, ywpi

* *** VIEWPORTS SPECIFIED BY AXIS ENDPOINTS ***
* Previous to V5.4 all viewports were specified as the corners of the PLOT+
* output window.  This made quick and dirty layouts very simple, but it
* meant an arduous process for someone to carefully lay out a multi-frame
* plot.
* In V5.4 we introduce DEFINE VIEW/AXES, in which the viewport position
* refers to the bounds of the active plotting area. The process of computing
* the corresponding PLOT+ output window is complex because the layout is a
* combination of normalized coordinates and PLOT+ "Denbo inches".
*
* The algorithm for determining the PLOT+ rectangle goes as follows.  It
* applies identically for X and Y.
*
*     |-------------------------|
*     |                         |
*     |    |---------------|CV  |
*     |    |               |    |
*     |    |   +++++++++CA |    |
*     |    |   +       +   |    |
*     |    |-OE-       -CE-|    |
*     |    |   +       +   |    |
*     |    |   +       +   |    |
*     |    | OA+++++++++   |    |
*     |    |      WA       |    |
*     |  OV|---------------|    |
*     |           WV            |
*     |-------------------------|
*                 WP
*
* We begin with notation
* "O" indicates an "origin" (lower left of a rectangle)
* "C" indicates "clip" (upper right of a rectangle)
* "W" indicates a width
*
* "V" indicates relative to the viewport
* "A" indicates relative to the axes of the viewport
* "E" indicates an "edge" distance -- the gap between the viewport and axes
* "P" indicates relative to the "page" (the current output window or metafile)
*
* "i" indicates the value is in units of "Denbo inches"
*     if "i" is absent it is normalized
*
* The known quantities are
* WPi = scale*width  - width of the output page (with scale factor)
* CA, OA - the limits of the axes given by the user in the DEF VIEW/AXES command
* OEi, CEi - the PLOT+ origin (1.2,1.4) and white space above the axes (1.0,1.4)
*
* We shift to normalized units using
*   OE = OEi/WPi
*   CE = CEi/WPi
*
* Giving
*   OV = OA - OE
*      = OA - OEi/WPi
* and similarly
*   CV = CA + CEi/WPi
*
* Determining the desired PLOT+ output window is further complicated by the
* ability of DEF VIEW/AXES to specify a viewport that implies a contradictory
* layout with respect to the "white space" surrounding the active plot
* rectangle.  For example, specifying an origin for the axes at 0,0 on the
* output window leaves no room for white space.  Thus in this code we adjust
* the PLOT+ origin as necessary to avoid this contradictory layout.

* ... V5.22 default behaviors (viewport not specified by axis)
        xWPi = scale*width
        yWPi = scale*height
        xoei = xwhitelo  ! ppl xorigin
        yoei = ywhitelo  ! ppl yorigin
        xcei = xwhitehi  ! white space to right of X axis
        ycei = ywhitehi  ! white space for docmtn at top of plot

        xov = xloax
        xcv = xhiax
        yov = yloax
        ycv = yhiax

        xwpi = scale * width
        ywpi = scale * height

        xcv = xcv + xcei/xwpi
        ycv = ycv + ycei/ywpi
        xov = xov - xoei/xwpi
        yov = yov - yoei/ywpi

* ... make adjustments if the normalized limits exceed allowable
        IF (xcv .GT. 1.0001) THEN
          xcei = xcei + (1.0-xcv)*xwpi
          xcv = 1.0
        ENDIF
        IF (ycv .GT. 1.0001) THEN
          ycei = ycei + (1.0-ycv)*ywpi
          ycv = 1.0
        ENDIF
        IF (xov .LT. -0.0001) THEN
          xoei = xoei + xov*xwpi
          xov = 0.0
        ENDIF
        IF (yov .LT. -0.0001) THEN
          yoei = yoei + yov*ywpi
          yov = 0.0
        ENDIF

* results
        xlovp = xov
        ylovp = yov
        xhivp = xcv
        yhivp = ycv

* success
	RETURN
	END
