	SUBROUTINE SET_AX_SIZES( xfrac, yfrac )

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
