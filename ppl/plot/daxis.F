      SUBROUTINE DAXIS(XF,YF,NMX,NMY)

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
	include 'parampl5_dat.decl'
	include 'PARAMPL5.DAT'
	include 'axis_inc.decl'
	include 'AXIS.INC'
	include 'labels_inc.decl'
	include 'LABELS.INC'
	include 'labcom_inc.decl'
	include 'LABCOM.INC'
	include 'hab_inc.decl'
	include 'HAB.INC'
	include 'lines_inc.decl'
	include 'LINES.INC'
	include 'pen_inc.decl'
	include 'PEN.INC'
	include 'pltl_inc.decl'
	include 'PLTL.INC'
	include 'mplot_inc.decl'
	include 'MPLOT.INC'
	include 'taxis_inc.decl'
	include 'TAXIS.INC'
	include 'tics_inc.decl'
	include 'TICS.INC'
	include 'tics2_inc.decl'
	include 'TICS2.INC'
	include 'axisl_inc.decl'
	include 'AXISL.INC'
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
	include 'xinterrupt.cmn'
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
