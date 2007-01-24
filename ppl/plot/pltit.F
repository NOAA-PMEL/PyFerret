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

*****************************************************

* local variable declarations for PLTIT (for TMAP)
	REAL	DIFF, x, xf, yf, xl, yl, xh, yh, xti, xtl, thi
	REAL*8	XLPLOT,YLPLOT,XHPLOT,YHPLOT
	REAL	xtm,ytm,xtx,ytx,minx,maxx,miny,maxy
	INTEGER nxs, nmx, nmy, ivloc, ip1, ip2, ip3, ip4, a_line

* TMAP include files with explicit type declarations:
	include 'parampl5_dat.decl'
	include 'PARAMPL5.DAT'

	DIMENSION X(NXS) ! Was VIRTUAL, changed for linux *jd*
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
	include 'mplot_inc.decl'
	include 'MPLOT.INC'
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

* special COMMON to let this routine know if FERRET is in control

	include 'ppl_in_ferret.cmn'
	include 'aline.cmn'	! jd for aline call
	include 'gkscm2.cmn' 
	include 'shade_vars.cmn' 


	LOGICAL CLEAR,CONTR,no_vpt_flag

	LOGICAL ITS_META_GRAPHICS, ITS_BATCH_GRAPHICS
        LOGICAL its_meta, batch, meta_batch

*	*jd* linux port 12.96 Add external stmt below
	external size, scale

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

        its_meta = ITS_META_GRAPHICS()
        batch = ITS_BATCH_GRAPHICS()
        meta_batch = batch .AND. its_meta

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
	CLEAR = .NOT.SOVER .AND. IMCNT.LE.1 .AND. .NOT.PLTOPN
C	SETUP AXIS FOR ALL PLOT TYPES.

*below	no_vpt_flag = .not.(vpt_was_set .and. ppl_viewports_active)!12.1.89 JD
*	MOD TO PROTECT COLORS ALREADY USED IN SHADE & FILL - took out for now
*	if (clear .and. no_vpt_flag) then
*	   protect_used_colors = .false.
*	   lo_color_ndx = 0
*	else
*	   protect_used_colors = .true.
*	endif

	if ( 
     .      ((.not.meta_actv).AND. (.not.meta_batch) ) .or.
     .      first_meta) goto 10		!12.1.89 JD

*	VPT_WAS_SET is used to check whether viewports were set the last time
*	Ferret made a plot call. It is however cleared, when CLSPLT is invoked,
*	as the continuity of viewport use is broken.

	no_vpt_flag = .not.(vpt_was_set .and. ppl_viewports_active)!12.1.89 JD
	if (clear .and. no_vpt_flag) then			!12.1.89 JD

	   if (.not. meta_clsplt) then				!1.25.90 JD
              call gdawk (meta_wsid)     			!11.6.89 JD
	      call gclwk (meta_wsid)				!11.6.89 JD
	      close (meta_lun)					!11.6.89 JD
*             TEMP FIX DUE TO GKS BUG  *JD* 5.91
* ----        close (aux_lun)
	   end if						!11.6.89 JD

	   call open_metafile					!11.6.89 JD
	   call gacwk (meta_wsid)				!11.6.89 JD
#ifndef xgks
	   call meta_version					!2.24.93
#endif
	   IF (.NOT. meta_batch) meta_actv = .true.					!11.6.89 JD
	   meta_clsplt = .false.				!1.25.90 JD
	end if							!11.6.89 JD
10      if (meta_actv.OR. meta_batch) then ! Put both following stmts in if section
           first_meta = .false.                                 !12.1.89 JD
           vpt_was_set = ppl_viewports_active                   !12.1.89 JD
        end if              ! to fix viewport metafile bug 11.4.93
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
	CALL COLOR(IPEN(0))
	IF(.NOT.SOVER)THEN
	    CALL SETAX(X,NXS,XF,YF,XL,YL,XH,YH,NMX,NMY,XTI,XTL)
	    CALL DAXIS(XF,YF,NMX,NMY)

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
	ELSE
	    IF(ITFLG.NE.0)THEN
		TLO = 1. - DIFF(IT1,ITSTRT)/DT
	        THI=TLO+DIFF(IT1,IT2)/DT
	    ENDIF
	    IF(ITFLG.EQ.1)THEN
		XTI=XHI
		XTL=XLO
		XHI=THI
		XLO=TLO
	    ELSE IF(ITFLG.EQ.-1)THEN
		XTI=YHI
		XTL=YLO
		YHI=THI
		YLO=TLO
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
*!	    IF(MAKEP.EQ.'AREA') CALL DAXIS(XF,YF,NMX,NMY) NAME CHANGE *jd*

C  Call WINDOW to restore values of XMIN, XMAX, YMIN, YMAX. Plot axis again
C  so any tics inside the box show up.  

	    IF(MAKEP.EQ.'FILL') THEN
               IF(IWIND.GT.0)CALL WINDOW(XLO,YLO,XHI,YHI)
               CALL DAXIS(XF,YF,NMX,NMY)
	       CALL DAXIS_BOX
            ENDIF


	ELSE IF(MAKEP.EQ.'PLOT') THEN
	   CALL PLOTXY(X,NXS,XF,YF)
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
****
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
	    CALL VELVCT(X,NXS,XF,YF)
		IF ( interrupted ) GOTO 3333 !*jd* 8.3.92
	ELSE IF(MAKEP.EQ.'PLOTV'.OR.MAKEP.EQ.'PLOTUV')THEN
	    CALL PLOTV(X,NXS,XF,YF)
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
	    CALL DAXIS(XF,YF,NMX,NMY)
	    CALL DAXIS_BOX
***************************************************
* TMAP addition for PLOTPOL call
	ELSE IF(MAKEP.EQ.'PLOTPOL')THEN
	    CALL PLOTPOL(X,NXS,XF,YF)
	    IF ( interrupted ) GOTO 3333
***************************************************
* TMAP addition for FILLPOL call
	ELSE IF(MAKEP.EQ.'FILLPOL' .OR. MAKEP.EQ.'POLYGON')THEN
	    CALL FILLPOL(X,NXS,XF,YF,iwind)
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

	    CALL DAXIS(XF,YF,NMX,NMY)
	    CALL DAXIS_BOX
***************************************************
	ENDIF

 3333	CONTINUE					! 1/89 - *sh*
* post-interrupt re-entry				! 1/89 - *sh*
	CALL SCALE(XF,YF,XORG,YORG,XLO,YLO)
C
	CALL WINDOW(0.,0.,0.,0.)
C
	IF ( .NOT.interrupted ) CALL DAXIS2(XF,YF,NMX,NMY) ! 1/89 - *sh*
	IF(ITFLG.EQ.1)THEN
	    XHI=XTI
	    XLO=XTL
	ELSE IF(ITFLG.EQ.-1)THEN
	    YHI=XTI
	    YLO=XTL
	ENDIF
*************************************
* TMAP addition: close segment and possibly write "** INTERRUPTED **" on plot
	IF ( PPL_in_FERRET ) CALL SEG_OFF
	IF ( interrupted ) CALL INTERRUPT( *5000 )	! 1/89 - *sh*
 4999	RETURN
 5000	CALL SAY_INTERRUPTED
*************************************
	RETURN
	END
