	SUBROUTINE PLTIT(X,NXS)
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

*****************************************************

* local variable declarations for PLTIT (for TMAP)
	REAL	DIFF, x, xf, yf, xl, yl, xh, yh, xti, xtl, thi
	real	xtm,ytm,xtx,ytx,minx,maxx,miny,maxy
	INTEGER nxs, nmx, nmy, ivloc, ip1, ip2, ip3, a_line
* TMAP include files with explicit type declarations:
#ifdef unix
	include 'tmap_pplv11inc/parampl5_dat.decl'
	include 'pplv11inc/PARAMPL5.DAT'
#else
	INCLUDE 'TMAP_PPLV11INC:PARAMPL5_DAT.DECL'
	INCLUDE 'PPLV11INC:PARAMPL5.DAT'
#endif
	DIMENSION X(NXS) ! Was VIRTUAL, changed for linux *jd*
C	KTYPE = 1 -XY  0 -Z  2-VECTOR
#ifdef unix
	include 'tmap_pplv11inc/lines_inc.decl'
	include 'pplv11inc/LINES.INC'
	include 'tmap_pplv11inc/pen_inc.decl'
	include 'pplv11inc/PEN.INC'
	include 'tmap_pplv11inc/axis_inc.decl'
	include 'pplv11inc/AXIS.INC'
	include 'tmap_pplv11inc/hd_inc.decl'
	include 'pplv11inc/HD.INC'
	include 'tmap_pplv11inc/taxis_inc.decl'
	include 'pplv11inc/TAXIS.INC'
	include 'tmap_pplv11inc/axisl_inc.decl'
	include 'pplv11inc/AXISL.INC'
	include 'tmap_pplv11inc/plt_inc.decl'
	include 'pplv11inc/PLT.INC'
	include 'tmap_pplv11inc/pltl_inc.decl'
	include 'pplv11inc/PLTL.INC'
	include 'tmap_pplv11inc/mplot_inc.decl'
	include 'pplv11inc/MPLOT.INC'
	include 'tmap_pplv11inc/cmrd_inc.decl'
	include 'pplv11inc/CMRD.INC'
	include 'tmap_pplv11inc/prmcom_inc.decl'
	include 'pplv11inc/PRMCOM.INC'
	include 'tmap_pplv11inc/switch_inc.decl'
	include 'pplv11inc/SWITCH.INC'
	include 'tmap_pplv11inc/system_inc.decl'
	include 'pplv11inc/SYSTEM.INC'
	include 'tmap_pplv11inc/cont_inc.decl'  !added to get ivcomp v1.1
	include 'pplv11inc/CONT.INC'  !added to get ivcomp v1.1
#else
	INCLUDE 'TMAP_PPLV11INC:LINES_INC.DECL'
	INCLUDE 'PPLV11INC:LINES.INC'
	INCLUDE 'TMAP_PPLV11INC:PEN_INC.DECL'
	INCLUDE 'PPLV11INC:PEN.INC'
	INCLUDE 'TMAP_PPLV11INC:AXIS_INC.DECL'
	INCLUDE 'PPLV11INC:AXIS.INC'
	INCLUDE 'TMAP_PPLV11INC:HD_INC.DECL'
	INCLUDE 'PPLV11INC:HD.INC'
	INCLUDE 'TMAP_PPLV11INC:TAXIS_INC.DECL'
	INCLUDE 'PPLV11INC:TAXIS.INC'
	INCLUDE 'TMAP_PPLV11INC:AXISL_INC.DECL'
	INCLUDE 'PPLV11INC:AXISL.INC'
	INCLUDE 'TMAP_PPLV11INC:PLT_INC.DECL'
	INCLUDE 'PPLV11INC:PLT.INC'
	INCLUDE 'TMAP_PPLV11INC:PLTL_INC.DECL'
	INCLUDE 'PPLV11INC:PLTL.INC'
	INCLUDE 'TMAP_PPLV11INC:MPLOT_INC.DECL'
	INCLUDE 'PPLV11INC:MPLOT.INC'
	INCLUDE 'TMAP_PPLV11INC:CMRD_INC.DECL'
	INCLUDE 'PPLV11INC:CMRD.INC'
	INCLUDE 'TMAP_PPLV11INC:PRMCOM_INC.DECL'
	INCLUDE 'PPLV11INC:PRMCOM.INC'
	INCLUDE 'TMAP_PPLV11INC:SWITCH_INC.DECL'
	INCLUDE 'PPLV11INC:SWITCH.INC'
	INCLUDE 'TMAP_PPLV11INC:SYSTEM_INC.DECL'
	INCLUDE 'PPLV11INC:SYSTEM.INC'
	INCLUDE 'TMAP_PPLV11INC:CONT_INC.DECL'  !Added to get IVCOMP v1.1
	INCLUDE 'PPLV11INC:CONT.INC'  !Added to get IVCOMP v1.1
#endif

#ifdef unix
	include 'tmap_pplv11inc/xinterrupt.cmn'	! 1/89 *sh* - interrupts
#else
	INCLUDE 'TMAP_PPLV11INC:XINTERRUPT.CMN'	! 1/89 *sh* - interrupts
#endif

* special COMMON to let this routine know if FERRET is in control
#ifdef unix
	include 'tmap_pplv11inc/ppl_in_ferret.cmn'
	include 'tmap_pplv11inc/aline.cmn'	! jd for aline call
	include 'tmap_pplv11inc/gkscm2.cmn' 
	include 'tmap_pplv11inc/shade_vars.cmn' 
#else
	INCLUDE 'TMAP_PPLV11INC:PPL_IN_FERRET.CMN'
	INCLUDE 'tmap_pplv11inc:tmap_aline.inc'	! JD For ALINE call
	INCLUDE 'tmap_pplv11inc:tmap_gkscm2.inc' 
	INCLUDE 'tmap_pplv11inc:tmap_shade_vars.inc' 
#endif

!	INCLUDE 'pplinc:PARAMPL5.DAT'
!	VIRTUAL X(NXS)
!	INCLUDE 'pplinc:LINES.INC'
!	INCLUDE 'pplinc:PEN.INC'
!	INCLUDE 'pplinc:AXIS.INC'
!    	INCLUDE 'pplinc:HD.INC'
!	INCLUDE 'pplinc:TAXIS.INC'
!	INCLUDE 'pplinc:AXISL.INC'
!	INCLUDE 'pplinc:PLT.INC'
!	INCLUDE 'pplinc:PLTL.INC'
!	INCLUDE 'pplinc:MPLOT.INC'
!	INCLUDE 'pplinc:CMRD.INC'
!	INCLUDE 'PPLINC:PRMCOM.INC'
!	INCLUDE 'cmdinc:SWITCH.INC'
!	INCLUDE 'cmdinc:SYSTEM.INC'


	LOGICAL CLEAR,CONTR,no_vpt_flag

*	*jd* linux port 12.96 Add external stmt below
	external size, scale

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC


	IF(((MAKEP.EQ.'CONTOUR'.OR.MAKEP.EQ.'VIEW'.or.
     *     makep.eq.'FILL'.or.makep.eq.'PIXEL'.or.
     *       makep.eq.'SHADE')  
     *	  .AND.JTYPE.EQ.1).OR.
     *	  ((MAKEP.EQ.'VECTOR').AND.JTYPE.GT.-1).OR.
     *	  ((MAKEP.EQ.'PLOT'.OR.MAKEP.EQ.'PLOTV'.OR.
     *	  MAKEP.EQ.'PLOTUV'.OR.MAKEP.EQ.'VELVCT'.OR.
     *	  MAKEP.EQ.'PLOTPOL'.OR.MAKEP.EQ.'FILLPOL')
     *	  .AND.JTYPE.NE.1))RETURN
	CONTR=MAKEP.EQ.'CONTOUR'.OR.MAKEP.EQ.'VIEW'
     *		.OR.MAKEP.EQ.'VECTOR'.or.makep.eq.'SHADE'  
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

	if (.not. meta_actv .or. first_meta) goto 10		!12.1.89 JD

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
	   meta_actv = .true.					!11.6.89 JD
	   meta_clsplt = .false.				!1.25.90 JD
	end if							!11.6.89 JD
10      if (meta_actv) then ! Put both following stmts in if section
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
		XL=(xtm-XLO)*XF
		XH=XLEN+(xtx-XHI)*XF
		YL=(ytm-YLO)*YF
		YH=YLEN+(ytx-YHI)*YF
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
	    CALL PLOTZ(X(IVLOC),XF,YF,XL,YL,XH,YH)
*!	    IF(MAKEP.EQ.'AREA') CALL DAXIS(XF,YF,NMX,NMY) NAME CHANGE *jd*
	    IF(MAKEP.EQ.'FILL') CALL DAXIS(XF,YF,NMX,NMY)
	    IF(MAKEP.EQ.'FILL') CALL DAXIS_BOX


	ELSE IF(MAKEP.EQ.'PLOT')THEN
	    CALL PLOTXY(X,NXS,XF,YF)
	    if (interrupted) go to 3333
**** JD * ALINE CALL 
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
		call aline_sub (x,nxs,a_line,minx,miny,maxx,maxy)
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
            ELSE IF(JTYPE.EQ.-2.AND.ivcomp.eq.2) then
                IVLOC=NX*NY+NX+NY+1
            ELSE
                IVLOC=1
            ENDIF
            call shade(x(ivloc))
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
	ELSE IF(MAKEP.EQ.'FILLPOL')THEN
	    CALL FILLPOL(X,NXS,XF,YF)
	    IF ( interrupted ) GOTO 3333
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
