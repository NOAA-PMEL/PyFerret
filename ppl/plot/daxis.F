      SUBROUTINE DAXIS(XF,YF,NMX,NMY)

* revised for TMAP
* from PPLUS :
*	Directory PMEL:[PLOT5.PPLV11.PLOT]DAXIS.FOR

* search for "TMAP" to find all revisions
* *sh*  - 1/23/89 added interrupt detection from interrupts in lower routines
*	  	  no explicit declarations - use FORT/WARN=NODECLARATIONS
C**
C**    @(#)daxis.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
#ifdef unix
	include 'tmap_pplv11inc/parampl5_dat.decl'
	include 'pplv11inc/PARAMPL5.DAT'
	include 'tmap_pplv11inc/axis_inc.decl'
	include 'pplv11inc/AXIS.INC'
	include 'tmap_pplv11inc/labels_inc.decl'
	include 'pplv11inc/LABELS.INC'
	include 'tmap_pplv11inc/labcom_inc.decl'
	include 'pplv11inc/LABCOM.INC'
	include 'tmap_pplv11inc/hab_inc.decl'
	include 'pplv11inc/HAB.INC'
	include 'tmap_pplv11inc/lines_inc.decl'
	include 'pplv11inc/LINES.INC'
	include 'tmap_pplv11inc/pen_inc.decl'
	include 'pplv11inc/PEN.INC'
	include 'tmap_pplv11inc/pltl_inc.decl'
	include 'pplv11inc/PLTL.INC'
	include 'tmap_pplv11inc/mplot_inc.decl'
	include 'pplv11inc/MPLOT.INC'
	include 'tmap_pplv11inc/taxis_inc.decl'
	include 'pplv11inc/TAXIS.INC'
	include 'tmap_pplv11inc/tics_inc.decl'
	include 'pplv11inc/TICS.INC'
	include 'tmap_pplv11inc/tics2_inc.decl'
	include 'pplv11inc/TICS2.INC'
	include 'tmap_pplv11inc/axisl_inc.decl'
	include 'pplv11inc/AXISL.INC'
#else
	INCLUDE 'TMAP_PPLV11INC:PARAMPL5_DAT.DECL'
	INCLUDE 'PPLV11INC:PARAMPL5.DAT'
	INCLUDE 'TMAP_PPLV11INC:AXIS_INC.DECL'
	INCLUDE 'PPLV11INC:AXIS.INC'
	INCLUDE 'TMAP_PPLV11INC:LABELS_INC.DECL'
	INCLUDE 'PPLV11INC:LABELS.INC'
	INCLUDE 'TMAP_PPLV11INC:LABCOM_INC.DECL'
	INCLUDE 'PPLV11INC:LABCOM.INC'
	INCLUDE 'TMAP_PPLV11INC:HAB_INC.DECL'
	INCLUDE 'PPLV11INC:HAB.INC'
	INCLUDE 'TMAP_PPLV11INC:LINES_INC.DECL'
	INCLUDE 'PPLV11INC:LINES.INC'
	INCLUDE 'TMAP_PPLV11INC:PEN_INC.DECL'
	INCLUDE 'PPLV11INC:PEN.INC'
	INCLUDE 'TMAP_PPLV11INC:PLTL_INC.DECL'
	INCLUDE 'PPLV11INC:PLTL.INC'
	INCLUDE 'TMAP_PPLV11INC:MPLOT_INC.DECL'
	INCLUDE 'PPLV11INC:MPLOT.INC'
	INCLUDE 'TMAP_PPLV11INC:TAXIS_INC.DECL'
	INCLUDE 'PPLV11INC:TAXIS.INC'
	INCLUDE 'TMAP_PPLV11INC:TICS_INC.DECL'
	INCLUDE 'PPLV11INC:TICS.INC'
	INCLUDE 'TMAP_PPLV11INC:TICS2_INC.DECL'
	INCLUDE 'PPLV11INC:TICS2.INC'
	INCLUDE 'TMAP_PPLV11INC:AXISL_INC.DECL'
	INCLUDE 'PPLV11INC:AXISL.INC'
#endif
	CHARACTER IFX*20,IFY*20

* TMAP interrupt flag
#ifdef unix
	include 'tmap_pplv11inc/xinterrupt.cmn'
#else
	INCLUDE 'tmap_pplv11inc:tmap_xinterrupt.cmn'
#endif

	IF(IMCNT.LE.1)THEN
	    CALL BOX
	END IF
C	SETUP AXIS
	CALL SETLAB(XF,YF,LTT,LTB,LYL,LYR,LXT,LXB,IFX,IFY)
C
C     DRAW AXIS, LEFT, RIGHT Y AXIS THEN TOP, BOTTOM XAXIS
C
	IF(MAKEP.EQ.'VIEW')GOTO 1000
      IF(MYLEF.EQ.0.AND.IAXON(3).EQ.1)THEN
	  IY=ITY
C
C	  YAXIS1 = LINEAR ,  YAXIS2 = LOG
C
          IF(ITFLG.EQ.-1)THEN
              CALL TAYIS(ITYPET,TLO,IT1,IT2,DT,NMTCT,LTB,IFDB,ISDB,
     *        YF,XLO,XF,TCSIZE)
	  ELSE IF(ITYPEY.EQ.1)THEN
	CALL YAXIS1(XLO,XF,YLO,YHI,YTIC,NMY,LINTY,1.,LYL,IFY,YF,
     *	    YCSIZE)
	  ELSE
            CALL YAXIS2(XLO,XF,YLO,YHI,0.,LYL,IFY,1,YF,YCSIZE)
          END IF
      END IF
	IF ( interrupted ) GOTO 3333		! TMAP interrupt
      IF(IAXON(4).EQ.1)THEN
	IY=-ITY
        IF(ITFLG.EQ.-1)THEN
            CALL TAYIS(ITYPET,TLO,IT1,IT2,DT,NMTCT,LTT,IFDB,ISDB,
     *      YF,XHI,XF,TCSIZE)
	ELSE IF(ITYPEY.EQ.1)THEN
	    CALL YAXIS1(XHI,XF,YLO,YHI,YTIC,NMY,LINTY,1.,LYR,IFY,YF,
     1     YCSIZE)
	ELSE
	    CALL YAXIS2(XHI,XF,YLO,YHI,0.,LYR,IFY,ITYPEY-1,YF,YCSIZE)
	ENDIF
      ENDIF
	IF ( interrupted ) GOTO 3333		! TMAP interrupt
      IF(IAXON(1).EQ.1)THEN
          IX=-ITX
C
C	  XAXIS1 = LINEAR, XAXIS2 = LOG, TAXIS CALLS EITHER TAXIS0,
C					 TAXIS1 OR TAXIS3
C
          IF(ITFLG.EQ.1)THEN
              CALL TAXIS(ITYPET,TLO,IT1,IT2,DT,NMTCT,LTT,IFDB,ISDB,
     *        XF,YHI,YF,TCSIZE)
          ELSE IF (ITYPEX.EQ.1)THEN
	      CALL XAXIS1(XLO,XHI,XTIC,NMX,LINTX,1.,LXT,IFX,XF,YHI,YF,
     *	       XCSIZE)
          ELSE
	      CALL XAXIS2(XLO,XHI,0.,LXT,IFX,ITYPEX-1,XF,YHI,YF,XCSIZE)
          END IF
      END IF
	IF ( interrupted ) GOTO 3333		! TMAP interrupt
      IF(MXBOT.EQ.0.AND.IAXON(2).EQ.1)THEN
          IX=ITX
          IF(ITFLG.EQ.1)THEN
              CALL TAXIS(ITYPET,TLO,IT1,IT2,DT,NMTCT,LTB,IFDB,ISDB,
     *        XF,YLO,YF,TCSIZE)
          ELSE IF (ITYPEX.EQ.1) THEN
	      CALL XAXIS1(XLO,XHI,XTIC,NMX,LINTX,1.,LXB,IFX,XF,YLO,YF,
     *	      XCSIZE)
          ELSE
	      CALL XAXIS2(XLO,XHI,0.,LXB,IFX,1,XF,YLO,YF,XCSIZE)
          END IF
      END IF

* TMAP interrupt re-entry
 3333	CONTINUE

	IF(IWIND.GT.0)CALL WINDOW(XLO,YLO,XHI,YHI)
	CALL CROSS
	CALL WINDOW(0.,0.,0.,0.)
C
C	DRAW AXIS LABELS
C
1000	CALL LABDRW(XF,YF,IFX,IFY)
	RETURN
	END
