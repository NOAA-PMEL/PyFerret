	SUBROUTINE SET_VP_SYMBOLS( scale, vp_size, xfrac, yfrac,
     .			           xspc_lo, yspc_lo, xspc_hi, yspc_hi,
     .                             xov,  yov,  xcv,  ycv)

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
* Set additional viewport symbols. Here is what happens with the scaling
* 11/30/04 change names of margin symbols.

ccc    DEFINE VIEWPORT/XLIM=xlo:xyi/YLIM=ylo:yhi[/TEXT=n/ my_view
ccc
ccc     vp_size(ivp) = value given by DEF VIEW/TEXT=      ! See xeq_define.F 
ccc                    or, SQRT( (xhi-xlo)) * (yhi-ylo) )
ccc
ccc	scale = 1.0 / SQRT( ABS(vp_size(vp_num)) )        ! See set_viewport.F
ccc                                                       ! if /TEXT=1, vp_size=1, scale=1
ccc
ccc     xfrac = scale*(xhi-xlo)
ccc     yfrac = scale*(yhi-ylo)

ccc	xsize = xfrac*width                               ! These are the viewport lengths 
ccc	ysize = yfrac*height                              ! reported by VP_WIDTH, VP_HEIGHT

* 11/16/2004 *acm*  from set_ax_sizes

	include 'plot_setup.parm'
c        include 'xplot_setup.cmn'      ! 10/91 for saved_ppl_yax
c        include 'PLT.INC'

* calling argument declarations:
        REAL scale, vp_size, xfrac, yfrac,
     .       xspc_lo, yspc_lo, xspc_hi, yspc_hi,
     .       xov,  yov,  xcv,  ycv

* internal variable declarations:
	INTEGER slen
	CHARACTER TM_FMT*8, buff*24

	buff = TM_FMT(scale,-3,8,slen) 
	CALL PPLCMD ( from, line, 0, 'SET VP_SCALE '//buff, 1, 1 )
 
c	buff = TM_FMT(vp_size,-3,8,slen) 
c	CALL PPLCMD ( from, line, 0, 'SET VP_SIZE '//buff, 1, 1 )

c	buff = TM_FMT(xfrac,-3,8,slen) 
c	CALL PPLCMD ( from, line, 0, 'SET VP_XSC '//buff, 1, 1 )
	
c	buff = TM_FMT(yfrac,-3,8,slen) 
c	CALL PPLCMD ( from, line, 0, 'SET VP_YSC '//buff, 1, 1 )
        
c	buff = TM_FMT(xspc_lo,-3,8,slen) 
c	CALL PPLCMD ( from, line, 0, 'SET VP_XORG '//buff, 1, 1 )
        
	buff = TM_FMT(xspc_hi,-3,8,slen) 
	CALL PPLCMD ( from, line, 0, 'SET VP_RT_MARGIN '//buff, 1, 1 )
        
c	buff = TM_FMT(yspc_lo,-3,8,slen) 
c	CALL PPLCMD ( from, line, 0, 'SET VP_YORG '//buff, 1, 1 )
        
	buff = TM_FMT(yspc_hi,-3,8,slen) 
	CALL PPLCMD ( from, line, 0, 'SET VP_TOP_MARGIN '//buff, 1, 1 )

	buff = TM_FMT(xov,-3,8,slen) 
	CALL PPLCMD ( from, line, 0, 'SET VP_XLO '//buff, 1, 1 )

	buff = TM_FMT(xcv,-3,8,slen) 
	CALL PPLCMD ( from, line, 0, 'SET VP_XHI '//buff, 1, 1 )

	buff = TM_FMT(yov,-3,8,slen) 
	CALL PPLCMD ( from, line, 0, 'SET VP_YLO '//buff, 1, 1 )

	buff = TM_FMT(ycv,-3,8,slen) 
	CALL PPLCMD ( from, line, 0, 'SET VP_YHI '//buff, 1, 1 )

	RETURN
	END
