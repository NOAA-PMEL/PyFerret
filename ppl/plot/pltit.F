 	SUBROUTINE PLTIT(X,NXS)
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
*  or commercial entity unless specific written per/x=178.5:179.5 uwndion is obtained
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
* revised for TMAP
* from PPLUS :
*	Directory PMEL:[PLOT5.LIBRARY.PLOT]
*	PLTIT.FOR;34              6  25-AUG-1987 15:23

* search for "TMAP" to find all revisions
* *sh* 6/30/88 - rev 1.0 - for FERRET 1.00+
*	- added calls to SEG_ON and SEG_OFF to imbed graphics in a segment
*	- this permits resizing of windows
* *jd* 9.7.88  - rev 2.0  
*	- Modified to call SHADE
* *jd* 11.28.88
*	- Modified to be compatible with version 1.1 PPL
* *sh*  - added interrupt detection and "in FERRET" detection
* *jd*  - added call to ALINE_SUB 10.17.89  (interrupts: 12.7.89)
* *jd*	- mod to handle metafiles 11.6.89
* *jd*	- mod to handle Ferret viewports 12.1.89
* *sh*  - mod to permit cancelling of all GKS output windows from Ferret
* *jd*  - mod to handle CLSPLT with metafiles
* *jd*  - mod to include AREA 1.14.92
* *jd*  - mod to add interrupts to PLOT calls
* *jd*  - mod to protect colors used already in shade plots 10.01.92
* *jd*  - mod to add PPLP version number in metafile
* *jd*  - mod to eliminate version number in xgks version
* *jd*  - mod to fix bug with WS not open error 11.4.93
* *jd*  - mod to fix bug in FILL, borders being overwritten and not redrawn,
*	  with call to daxis_box 8.18.95 - also use with SHADE
* *jd*	- mod to put back use of daxis as cross and interior tics are
*	  lost without it.
* *sh*  - added commands PLOTPOL and FILLPOL
* *jd*	- put in daxis after FILLPOL
* *acm*	- 10/2000 add VECTOR/FLOWLINE, alias FLOWLINE
* *acm* - 12/2001 send third argument to FLOWLINE, work array for the
*         flag grid, now allocated in disp_set_up.F
* *acm* - 10/02 After FILL plot (plotz) call WINDOW to restore values of
*         XMIN, XMAX, YMIN, YMAX. Else DAXIS plot calls are not in range.
* *acm* - 10/02 bug fix: start index of X as the third argument to FLOWLINE
*         was icorrect for irregular-grid case (JTYPE .ne. -1) See PPLLDC.
* *acm*   V550 11/02 For fine grids, XLPLOT, XHPLOT, YLPLOT, YHPLOT 
*         computations need to be done in double precision.
* *acm*   V550 12/02 For lines which are plotted with symbols, plot a 
*            key having 3 points.
* *acm*   v552  3/03 add word POLYGON so can say PPL FILLPOL, or PPL POLY
 
* V554  9/03 *acm* Compute XLPLOT etc in double precision: for fine
*                  grids in curvlinear coordinates; xtm, xtx are axis indices
*                  (like 4000) and xhi, xlo are in the final curvilinear 
*                  coordinates (like 0.0218) so adding xtm+xlo as R*4 loses 
*                  the precision in xtm. See also plotz.F, setax.F
* V602  1/07 *acm* true batch mode for metafiles; flag meta_batch
*                  distinguises this case from other batch modes.
* V6.11 4/08 *acm* Fixes for bug 1571
* *acm* 5/09 *acm* Fixes for bug 1663. When redrawing axes after a POLY/OVERLAY
*                  do not do any axis labels
* V6.5  3/10 *acm* When re-plotting axes, do not redraw the main label.
* v68  *acm* 4/08 Allow shade, vector plots of degenerate regions - world region 
*              within a grid cell
* PyFr  11/11 *kms* added call to consider updating the drawing when done
** V68   *acm* 4/11 change how flag meta_batch is computed (different Makefile)
* V6.8 *acm* 7/9/2012 For batch graphics on RHEL6, change ITS_BATCH_GRAPHICS
*                     from logical to integer, then batch = ITS_BATCH_GRAPHICS().NE.0
* V694  *acm*  5/15 Ticket 2283. Handling axis scaling for ribbon plots.
* V694  *acm*  1/15 Changes to allow two time axes on plot (F-T)
* V694  *acm*  2/15 Fix ticket 2232; redraw plot axes on line plots
* V694  *acm*  5/15 correct the setting of is_ribbon
* V695  *acm*  5/15 Ticket 2279: PyFerret don't redraw lines
* V695 *acm*   5/15 Ticket 2279: Draw axis lines, tics and labels only once.  It does not
*                   work to call DAXIS only at the end of this routine; the calls to axis
*                   subroutines apparently make some settings for scaling on the page.
*                   Call daxis with a flag "DRAW" set to false, which executes the 
*                   routine but does not make drawing calls.
* V695  *acm*  6/15 Ticket 2297, scaling on overlay plots
* V695  *acm*  9/15 Changes for hour:min labeling on time axes
* V697 *acm*  12/15 Ticket 2334: For time roundoff on overlays, changing 
*                   things only for the short-time-axis case.
* V 7 *acm*   6/16 Ticket 2435: moveable labels in PyFerret when using PyFerret fonts
* V73+ 1/18 *acm*  Changes for Issue 1009; precsision in time info sent to pplus for 2D plots
* V74  *acm* -11/18 Issue #1900 Fixes for tic marks and PLOT/OVERLAY with short-interval time
* v744 *acm* 11/18 Issue 1786: graticule lines & axes drawn if PLOT/OVER/AXES given
* v76  *acm*  4/20 New flag ribbon_on so that RIBBON/SET; PPL PLOT works correctly
* v761 *acm*  7/20 Issue 1982. Save XHI, XLO, YHI, YLO Needed for axes on an /OVER/AXES plot.
* V766  4/22 *acm* For Issue 1009, fixes here to allow sequences such as: 
*                   FILL/NOAXIS [var]; FILL/OVER/AXES [var]
*                   For some plots with a time axis, the time axes and/or the overlay 
*                   plot went missing due to scaling errors
*		

*****************************************************

* local variable declarations for PLTIT (for TMAP)
	REAL	x, xf, yf, xl, yl, xh, yh, xti, xtl, thi
	REAL*8	DIFF, XLPLOT,YLPLOT,XHPLOT,YHPLOT
	REAL	xtm,ytm,xtx,ytx,minx,maxx,miny,maxy
	INTEGER nxs, nmx, nmy, ivloc, ip1, ip2, ip3, ip4, a_line,
     .          LABELX_save, LABELY_save, LABELT_save
	LOGICAL DRAW, did_draw

* TMAP include files with explicit type declarations:
	include 'parampl5_dat.decl'
	include 'PARAMPL5.DAT'

	DIMENSION X(NXS) ! Was VIRTUAL, changed for linux *jd*
	include 'lines_inc.decl'
	include 'LINES.INC'
	include 'pen_inc.decl'
	include 'PEN.INC'
	include 'axis_inc.decl'
	include 'AXIS.INC'
	include 'hd_inc.decl'
	include 'HD.INC'
	include 'taxis_inc.decl'
	include 'TAXIS.INC'
	include 'axisl_inc.decl'
	include 'AXISL.INC'
	include 'plt_inc.decl'
	include 'PLT.INC'
	include 'pltl_inc.decl'
	include 'PLTL.INC'
	include 'cmrd_inc.decl'
	include 'CMRD.INC'
	include 'prmcom_inc.decl'
	include 'PRMCOM.INC'
	include 'switch_inc.decl'
	include 'SWITCH.INC'
	include 'system_inc.decl'
	include 'SYSTEM.INC'
	include 'cont_inc.decl'  !added to get ivcomp v1.1
	include 'CONT.INC'  !added to get ivcomp v1.1
	include 'xinterrupt.cmn'	! 1/89 *sh* - interrupts
	include 'labels_inc.decl'
	include 'LABELS.INC'
	include 'miss_inc.decl'
	include 'MISS.INC'


* special COMMON to let this routine know if FERRET is in control

	include 'ppl_in_ferret.cmn'
	include 'aline.cmn'	! jd for aline call
	include 'gkscm2.cmn' 
	include 'shade_vars.cmn' 


	LOGICAL CLEAR,CONTR,no_vpt_flag, is_ribbon
	REAL	fp_xfac, fp_yfac

*	*jd* linux port 12.96 Add external stmt below
	external size, scale

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC


	IF(((MAKEP.EQ.'CONTOUR'.OR.MAKEP.EQ.'VIEW'.or.
     *     makep.eq.'FILL'.or.makep.eq.'PIXEL'.or.
     *       makep.eq.'SHADE')  
     *	  .AND.JTYPE.EQ.1).OR.
     *	  ((MAKEP.EQ.'VECTOR').AND.JTYPE.GT.-1).OR.
     *	  ((MAKEP.EQ.'FLOWLINE').AND.JTYPE.GT.-1).OR.
     *	  ((MAKEP.EQ.'PLOT'.OR.MAKEP.EQ.'PLOTV'.OR.
     *	  MAKEP.EQ.'PLOTUV'.OR.MAKEP.EQ.'VELVCT'.OR.
     *	  MAKEP.EQ.'PLOTPOL'.OR.MAKEP.EQ.'FILLPOL')
     *	  .AND.JTYPE.NE.1))RETURN
	CONTR=MAKEP.EQ.'CONTOUR'.OR.MAKEP.EQ.'VIEW'
     *		.OR.MAKEP.EQ.'VECTOR'.or.makep.eq.'SHADE'   
     *          .or.makep.eq.'FLOWLINE'
     *          .or.makep.eq.'FILL'.or.makep.eq.'PIXEL'
	CLEAR = .NOT.SOVER .AND. .NOT.PLTOPN
C	SETUP AXIS FOR ALL PLOT TYPES.

*below	no_vpt_flag = .not.(vpt_was_set .and. ppl_viewports_active)!12.1.89 JD
*	MOD TO PROTECT COLORS ALREADY USED IN SHADE & FILL - took out for now
*	if (clear .and. no_vpt_flag) then
*	   protect_used_colors = .false.
*	   lo_color_ndx = 0
*	else
*	   protect_used_colors = .true.
*	endif

	IF(.NOT.SOVER .AND. .NOT.PLTOPN)CALL SIZE(WIDTH,HEIGHT)
#ifdef unix
*       call flush(lttout)
#endif
***********************************************
* TMAP check that output window is active
	IF ( PPL_in_FERRET ) CALL CK_GKS_ACTIVE( *4999 )
***********************************************
	CALL OPNDEV(TERMF,CLEAR)
***********************************************
* TMAP addition to turn on segment
	IF ( PPL_in_FERRET ) CALL SEG_ON( )                              
***********************************************
	IF (sover) THEN
	   xlosave = xlo
	   xhisave = xhi
	   ylosave = ylo
	   yhisave = yhi
	ENDIF

	CALL COLOR(IPEN(0))
	IF (MAKEP .EQ. 'PLOT'. AND.ribbon_on) MAKEP = 'RIBPLOT'
	IF(.NOT.SOVER)THEN
	    is_ribbon = MAKEP.EQ.'RIBPLOT'
	    CALL SETAX(X,NXS,XF,YF,XL,YL,XH,YH,NMX,NMY,XTI,XTL,is_ribbon)

	    DRAW = .TRUE.
	    IF (makep.EQ.'FILL' .OR. makep.EQ.'SHADE' .OR. 
     .          makep.EQ.'POLYGON' .OR. makep.EQ.'FILLPOL') DRAW = .FALSE.

	    IF (ABS(XF) .GT. CMZGE .OR. ABS(YF) .GT. CMZGE) GOTO 3333  ! fill/contour but 1 grid point
	    CALL DAXIS(XF,YF,NMX,NMY,DRAW)

		XLPLOT=DBLE(XL) 
		XHPLOT=DBLE(XH)
		YLPLOT=DBLE(YL)
		YHPLOT=DBLE(YH)

		call trans (0,xmin,ymin,xtm,ytm)
		call trans (0,xmax,ymax,xtx,ytx)

		XLPLOT=(DBLE(xtm)-DBLE(XLO) )*DBLE(XF) 
		XHPLOT=DBLE(XLEN)+(DBLE(xtx)-DBLE(XHI) )*DBLE(XF)
		YLPLOT=(DBLE(ytm)-DBLE(YLO) )*DBLE(YF)
		YHPLOT=DBLE(YLEN)+(DBLE(ytx)-DBLE(YHI) )*DBLE(YF)

	    IF ( interrupted ) GOTO 3333		! 1/89 - *sh*

	    IF (ITFLG.EQ.1) THEN
	       TLOX_save = XLO
	       THIX_save = XHI
	       DT_SAVE = dt
	    ENDIF
	    IF(ITFLG.EQ.-1 .OR. ITFLG.EQ.2) THEN
	       TLOY_save = YLO
	       THIY_save = YHI
	       DT_SAVE = dt
	    ENDIF

	ELSE
	    IF(ITFLG.NE.0)THEN
	        ! Handle time as in setax for short-time axes
		! Round off to the nearest minute 
	        
		IF (is_hours_axis .GT. 0) CALL FIXTIM (ITMIN,ITMAX,IT1,IT2,4)
		TLO = 1. - DIFF(IT1,ITSTRT)/DT
	        THI=TLO+DIFF(IT1,IT2)/DT

		IF (is_hours_axis .GT. 0) THEN
		   IF (ITFLG.EQ. 1) THEN
		      tlo = TLOX_save
		      thi = THIX_save
		   ENDIF
		   IF (ITFLG.EQ.-1 .OR. ITFLG.EQ.2) THEN
		      tlo = TLOY_save
		      thi = THIY_save
		   ENDIF
		ENDIF
	    ENDIF
	    IF(ITFLG.EQ.1 .OR. ITFLG.EQ.2)THEN
		XTI=XHI
		XTL=XLO
		XHI=THI
		XLO=TLO
		IF (contr) CALL CHECK_T_SCALE ('X', xlo, xhi)
	    ELSE IF(ITFLG.EQ.-1 .OR. ITFLG.EQ.2)THEN
		YTI=YHI
		YTL=YLO
		YHI=THI
		YLO=TLO
		IF (contr) CALL CHECK_T_SCALE ('Y', ylo, yhi)
	    ENDIF

	    IF(CONTR)THEN
		call trans (0,xmin,ymin,xtm,ytm)
		call trans (0,xmax,ymax,xtx,ytx)

		XLPLOT=(DBLE(xtm)-DBLE(XLO) )*DBLE(XF) 
		XHPLOT=DBLE(XLEN)+(DBLE(xtx)-DBLE(XHI) )*DBLE(XF)
		YLPLOT=(DBLE(ytm)-DBLE(YLO) )*DBLE(YF)
		YHPLOT=DBLE(YLEN)+(DBLE(ytx)-DBLE(YHI) )*DBLE(YF)

	    ENDIF

	ENDIF


C	DRAW AXIS AND LABELS
C	DO PLOTTING
C
	IF(IWIND.GT.0)CALL WINDOW(XLO,YLO,XHI,YHI)
C
	IF(MAKEP.EQ.'CONTOUR'.or.makep.eq.'FILL'.or.
     *     makep.eq.'PIXEL')THEN
	    IF(JTYPE.EQ.-1.AND.ivcomp.eq.2) then
	   	IVLOC=NX*NY+1
	    ELSE IF(JTYPE.EQ.-2.AND.ivcomp.eq.2) then
		IVLOC=NX*NY+NX+NY+1
	    ELSE
		IVLOC=1
	    ENDIF
	    CALL PLOTZ(X(IVLOC),XF,YF,XLPLOT,YLPLOT,XHPLOT,YHPLOT)

C  Call WINDOW to restore values of XMIN, XMAX, YMIN, YMAX. 

	    IF(MAKEP.EQ.'FILL') THEN
               IF(IWIND.GT.0)CALL WINDOW(XLO,YLO,XHI,YHI)
            ENDIF

	ELSE IF(MAKEP.EQ.'PLOT') THEN
	   IPV = 0
	   CALL PLOTXY(X,NXS,XF,YF,IPV)

	   if (interrupted) go to 3333

****    JD * ALINE CALL 
	   CALL WINDOW(0.,0.,0.,0.)
C
	    do 100 a_line = 1,lnum
		if (.not. aline_on(a_line)) goto 100

		if (aline_user(a_line)) then
		   minx = aline_minx(a_line)
		   miny = aline_miny(a_line)
		   maxx = aline_maxx(a_line)
		   maxy = aline_maxy(a_line)
		else
		   call point_convert (aline_minx(a_line),aline_miny(a_line),
     .		   minx,miny)
		   call point_convert (aline_maxx(a_line),aline_maxy(a_line),
     .		   maxx,maxy)
		end if

***  ACM 12/02 For lines which are plotted with symbols, plot a 
***            key having 3 points.

C		call aline_sub (x,nxs,a_line,minx,miny,maxx,maxy)
		call line_key (x,nxs,a_line,minx,miny,maxx,maxy)
		IF ( interrupted ) GOTO 3333 !*jd* 8.3.92
100	    continue

* See ticket 2565, shifted lbels on time overlay plots
	   IF(SOVER) THEN 
	     IF (ITFLG.EQ.1 .OR. ITFLG.EQ.2) THEN
	      XHI = XTI
	      XLO = XTL
	     ELSEIF (ITFLG.EQ.-1 .OR. ITFLG.EQ.-2) THEN
	      YHI = YTI
	      YLO = YTL
	     ENDIF
	   ENDIF

	ELSE IF(MAKEP.EQ.'RIBPLOT') THEN
	   CALL PLOTXY_RIBBON(X,NXS,XF,YF)
	   IF ( interrupted ) GOTO 3333

	ELSE IF(MAKEP.EQ.'VECTOR')THEN
	    IF(JTYPE.EQ.-1)THEN
		CALL VECFLD(X(1),X(NX*NY+1),NX,NY,X(1),X(1),XF,YF)
		IF ( interrupted ) GOTO 3333 !*jd* 8.3.92
	    ELSE
		IP1=NX*NY+1
		IP2=IP1+NX
		IP3=IP2+NY
  		CALL VECFLD(X(1),X(IP3),NX,NY,X(IP1),X(IP2),XF,YF)
		IF ( interrupted ) GOTO 3333 !*jd* 8.3.92
	    ENDIF
****
C ACM Add FLOWLINE call based on VECFLD

	ELSE IF(MAKEP.EQ.'FLOW')THEN
	    IF(JTYPE.EQ.-1)THEN
		CALL FLOWLINE(X(1),X(NX*NY+1),X(2*NX*NY+1),
     .            NX,NY,X(1),X(1),XF,YF)
		IF ( interrupted ) GOTO 3333 
	    ELSE
		IP1=NX*NY+1
		IP2=IP1+NX
		IP3=IP2+NY
                IP4=2*NX*NY + 2*(NX+NY)
  		CALL FLOWLINE(X(1),X(IP3),X(IP4),NX,NY,X(IP1),
     .             X(IP2),XF,YF)
		IF ( interrupted ) GOTO 3333
	    ENDIF
        ELSE IF(MAKEP.EQ.'VELVCT')THEN
c	    CALL VELVCT(X,NXS,XF,YF)
	    IPV = 2
	    CALL PLOTXY(X,NXS,XF,YF,IPV)
		IF ( interrupted ) GOTO 3333 !*jd* 8.3.92
	ELSE IF(MAKEP.EQ.'PLOTV'.OR.MAKEP.EQ.'PLOTUV')THEN
c	    CALL PLOTV(X,NXS,XF,YF)
	    IPV=1
	    CALL PLOTXY(X,NXS,XF,YF,IPV)

		IF ( interrupted ) GOTO 3333 !*jd* 8.3.92
	ELSE IF(MAKEP.EQ.'VIEW')THEN
	    IF(JTYPE.EQ.-1.AND.ivcomp.eq.2) then
	   	IVLOC=NX*NY+1
	    ELSE IF(JTYPE.EQ.-2.AND.ivcomp.eq.2) then
		IVLOC=NX*NY+NX+NY+1
	    ELSE
		IVLOC=1
	    ENDIF
	    CALL VIEW(X(IVLOC))
		IF ( interrupted ) GOTO 3333 !*jd* 8.3.92
***************************************************
* TMAP addition for SHADE call
        else if (makep .eq. 'SHADE') then
            IF(JTYPE.EQ.-1.AND.ivcomp.eq.2) then
                IVLOC=NX*NY+1
		IP1=NX*NY+1
		IP2=IP1+NX
		IP3=IP2+NY
                IP4=2*NX*NY + 2*(NX+NY)
            ELSE IF(JTYPE.EQ.-2.AND.ivcomp.eq.2) then
                IVLOC=NX*NY+NX+NY+1
            ELSE
                IVLOC=1
                IP4=NX*NY+1
            ENDIF
            call shade(x(ivloc), x(IP4) )
	    IF ( interrupted ) GOTO 3333		! 1/89 - *sh*

***************************************************
* TMAP addition for PLOTPOL call
	ELSE IF(MAKEP.EQ.'PLOTPOL')THEN

	   STOP  'PLOTPOL not yet implemented'
c	    CALL PLOTPOL(X,NXS,XF,YF)
	    IPV=3
	    CALL PLOTXY(X,NXS,XF,YF,IPV)
	    IF ( interrupted ) GOTO 3333
***************************************************
* TMAP addition for FILLPOL call
	ELSE IF(MAKEP.EQ.'FILLPOL' .OR. MAKEP.EQ.'POLYGON')THEN
* Get the time-axis scale right. These factors account for the fact
* that the scaling is set for the range of the polygon data but we are
* rescaling the overlay to be on the entire underlying time axis
	    fp_xfac = 1.
	    fp_yfac = 1.
	    IF (SOVER .AND. ITFLG.NE.0) THEN
	       IF  (ITFLG.EQ.1 .OR. ITFLG.EQ.2) THEN
	          fp_xfac = (THIX_save-TLOX_save)/(xhi-xlo)  ! expand range by this factor
		  XHI = THIX_save 
		  XLO = TLOX_save
	       ELSE IF (ITFLG.EQ.11 .OR. ITFLG.EQ.2) THEN
	          fp_yfac = (THIY_save-TLOY_save)/(yhi-ylo)
		  YHI = THIY_save 
		  YLO = TLOY_save
	       ENDIF
	    ENDIF
	    CALL FILLPOL(X,NXS,XF,YF,fp_xfac,fp_yfac)
	    IF ( interrupted ) GOTO 3333

	    if (sline) then
	       CALL WINDOW(0.,0.,0.,0.)
	       
*              Only one line at a time to label
	       do 200 a_line = 1,1
		  if (.not. aline_on(a_line)) goto 200
		  
		  if (aline_user(a_line)) then
		     minx = aline_minx(a_line)
		     miny = aline_miny(a_line)
		     maxx = aline_maxx(a_line)
		     maxy = aline_maxy(a_line)
		  else
		     call point_convert 
     .	               (aline_minx(a_line),aline_miny(a_line),
     .		        minx,miny)
		     call point_convert 
     .	               (aline_maxx(a_line),aline_maxy(a_line),
     .                  maxx,maxy)
		  end if
		  call aline_sub (x,nxs,a_line,minx,miny,maxx,maxy)
		  
		  IF ( interrupted ) GOTO 3333 !*jd* 8.3.92
 200	       continue
	    end if

***************************************************
	ENDIF

 3333	CONTINUE					! 1/89 - *sh*
* post-interrupt re-entry				! 1/89 - *sh*
	
* Draw the plot axes 

	IF (sover .AND. DRAW) THEN
	   xlo = xlosave
	   xhi = xhisave
	   ylo = ylosave
	   yhi = yhisave

	   IF (ITFLG.EQ.1) THEN
	       TLO = TLOX_save
	       THI = THIX_save
	       DT = DT_save
	   ENDIF
	   IF(ITFLG.EQ.-1 .OR. ITFLG.EQ.2) THEN
	       TLO = TLOY_save
	       THI = THIY_save
	       DT = DT_save
	   ENDIF
	ENDIF

	did_draw = draw
	IF (.NOT.SOVER .OR. axes_on_overlay) THEN
	   IF (axes_on_overlay) lab1 = ' '
	   DRAW = .FALSE.
	   CALL DAXIS(XF,YF,NMX,NMY,DRAW)
	   DRAW = (.NOT.did_draw .OR. axes_on_overlay)
	   CALL DAXIS(XF,YF,NMX,NMY,DRAW)
	   CALL SCALE(XF,YF,XORG,YORG,XLO,YLO)
	   axes_on_overlay = .FALSE.
	ENDIF
	
	IF (SOVER) lab1 = ' '
C
	CALL WINDOW(0.,0.,0.,0.)
C
C Put movable labels on
	IF ( .NOT.interrupted ) CALL DAXIS2(XF,YF,NMX,NMY) ! 1/89 - *sh*
	IF (SOVER) THEN 
	   IF(ITFLG.EQ.1 .OR. ITFLG.EQ.2)THEN
	       XHI=XTI
	       XLO=XTL
	   ELSE IF(ITFLG.EQ.-1 .OR. ITFLG.EQ.2)THEN
	       YHI=YTI
	       YLO=YTL
	   ENDIF
	ENDIF
*************************************
* TMAP addition: close segment and possibly write "** INTERRUPTED **" on plot
	IF ( PPL_in_FERRET ) CALL SEG_OFF
*       consider updating the drawing now
        CALL FGD_CONSIDER_UPDATE(.FALSE.)
*
	IF ( interrupted ) CALL INTERRUPT( *5000 )	! 1/89 - *sh*
 4999	RETURN
 5000	CALL SAY_INTERRUPTED
*************************************
	RETURN
	END
