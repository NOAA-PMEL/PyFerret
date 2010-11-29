      SUBROUTINE DAXIS(XF,YF,NMX,NMY,MAINLAB)

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
* v5.41 *acm* - 3/02  bug fix: xaxis2 call for log plots caused the tics for log 
*               axes to be wrong.
*               Also remove VMS include statements.
* v554: *acm* 2/04 add /HGRAT /VGRAT to draw optional graticule lines at tic marks:
*                  Pass new arguments to all axis routines, and set hline, vline
*                  when done.
C *acm* 11/05  new argument to xaxis1 and yaxis1 for deg/min/sec labels
C        1/06  xvaloff,yvaloff add this amount to each x or y axis label. 
C              (axis data was read as double precision and converted to single 
C              after applying an offset.)
C *acm* 5/09   Extra argument lets us skip the plot label (when redrawing
C              axes after a POLY/OVERLAY)
C *acm* 10/09  Fix bug 1708: YAXIS2 call for log plots caused the tics for the 
*              right hand side log axes to be wrong when we have a depth axis.
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

	CHARACTER IFX*20,IFY*20
        integer ii, MAINLAB

* TMAP interrupt flag
	include 'xinterrupt.cmn'

C If graticule lines are to be drawn across the plot at the tics
C then only do them for one horizontal and one vertical axis.  
C The dash lines interfere with each other if drawn from both 
C sides. HLINE and VLINE control this. Draw a second axis after each 
C axis that may have the graticule lines, so the axes themselves are 
C always solid, black.

        HLINE = .TRUE.
        VLINE = .TRUE.
           
        do ii = 1, 2   ! first time thru do graticules, 
                      ! second time dont, get black axes
        
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
C	  YAXIS1 = LINEAR ,  YAXIS2 = LOG
C
          IF(ITFLG.EQ.-1)THEN
              CALL TAYIS(ITYPET,TLO,IT1,IT2,DT,NMTCT,LTB,IFDB,ISDB,
     *        YF,XLO,XF,TCSIZE,XHI,HGRAT, HCOLOR, HLINE)
              CALL TAYIS(ITYPET,TLO,IT1,IT2,DT,NMTCT,LTB,IFDB,ISDB,
     *        YF,XLO,XF,TCSIZE,XHI,HGRAT, HCOLOR, .FALSE.)
            hline = .FALSE.
	  ELSE IF(ITYPEY.EQ.1)THEN
	   CALL YAXIS1(XLO,XF,YLO,YHI,YTIC,NMY,LINTY,1.,LYL,IFY,YF,
     *	    YCSIZE,XHI,YDMS,HGRAT, HCOLOR, HLINE, YVALOFF)
	   CALL YAXIS1(XLO,XF,YLO,YHI,YTIC,NMY,LINTY,1.,LYL,IFY,YF,
     *	    YCSIZE,XHI,YDMS,HGRAT, HCOLOR, .FALSE., YVALOFF)
            hline = .FALSE.
	  ELSE
            CALL YAXIS2(XLO,XF,YLO,YHI,0.,LYL,IFY,1,YF,YCSIZE,
     .                    XHI,HGRAT, HCOLOR, HLINE)
            CALL YAXIS2(XLO,XF,YLO,YHI,0.,LYL,IFY,1,YF,YCSIZE,
     .                    XHI,HGRAT, HCOLOR, .FALSE.)
            hline = .FALSE.
          END IF
      END IF
	IF ( interrupted ) GOTO 3333		! TMAP interrupt
      IF(IAXON(4).EQ.1)THEN
	IY=-ITY
        IF(ITFLG.EQ.-1)THEN
            CALL TAYIS(ITYPET,TLO,IT1,IT2,DT,NMTCT,LTT,IFDB,ISDB,
     *      YF,XHI,XF,TCSIZE,XLO,HGRAT, HCOLOR, HLINE)
            CALL TAYIS(ITYPET,TLO,IT1,IT2,DT,NMTCT,LTT,IFDB,ISDB,
     *      YF,XHI,XF,TCSIZE,XLO,HGRAT, HCOLOR, .FALSE.)
           hline = .FALSE.
	ELSE IF(ITYPEY.EQ.1)THEN
	    CALL YAXIS1(XHI,XF,YLO,YHI,YTIC,NMY,LINTY,1.,LYR,IFY,YF,
     1     YCSIZE,XLO,YDMS,HGRAT, HCOLOR, HLINE, YVALOFF)
	    CALL YAXIS1(XHI,XF,YLO,YHI,YTIC,NMY,LINTY,1.,LYR,IFY,YF,
     1     YCSIZE,XLO,YDMS,HGRAT, HCOLOR, .FALSE., YVALOFF)
           hline = .FALSE.
	ELSE
	   CALL YAXIS2(XHI,XF,YLO,YHI,0.,LYR,IFY,1,YF,YCSIZE,
     .                    XLO,HGRAT, HCOLOR, HLINE)
	   CALL YAXIS2(XHI,XF,YLO,YHI,0.,LYR,IFY,1,YF,YCSIZE,
     .                    XLO,HGRAT, HCOLOR, .FALSE.)
           hline = .FALSE.
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
     *        XF,YHI,YF,TCSIZE,YLO,VGRAT, VCOLOR, VLINE)
              CALL TAXIS(ITYPET,TLO,IT1,IT2,DT,NMTCT,LTT,IFDB,ISDB,
     *        XF,YHI,YF,TCSIZE,YLO,VGRAT, VCOLOR, .FALSE.)
              vline = .FALSE.
          ELSE IF (ITYPEX.EQ.1)THEN
	      CALL XAXIS1(XLO,XHI,XTIC,NMX,LINTX,1.,LXT,IFX,XF,YHI,YF,
     *	       XCSIZE,YLO,XDMS,VGRAT, VCOLOR, VLINE, XVALOFF)
	      CALL XAXIS1(XLO,XHI,XTIC,NMX,LINTX,1.,LXT,IFX,XF,YHI,YF,
     *	       XCSIZE,YLO,XDMS,VGRAT, VCOLOR, .FALSE., XVALOFF)
              vline = .FALSE.
          ELSE
	      CALL XAXIS2(XLO,XHI,0.,LXT,IFX,1,XF,YHI,YF,XCSIZE,
     *                    YLO,VGRAT, VCOLOR, VLINE)
	      CALL XAXIS2(XLO,XHI,0.,LXT,IFX,1,XF,YHI,YF,XCSIZE,
     *                    YLO,VGRAT, VCOLOR, .FALSE.)
              vline = .FALSE.
          END IF
      END IF
	IF ( interrupted ) GOTO 3333		! TMAP interrupt
      IF(MXBOT.EQ.0.AND.IAXON(2).EQ.1)THEN
          IX=ITX
          IF(ITFLG.EQ.1)THEN
              CALL TAXIS(ITYPET,TLO,IT1,IT2,DT,NMTCT,LTB,IFDB,ISDB,
     *        XF,YLO,YF,TCSIZE,YHI,VGRAT, VCOLOR, VLINE)
              CALL TAXIS(ITYPET,TLO,IT1,IT2,DT,NMTCT,LTB,IFDB,ISDB,
     *        XF,YLO,YF,TCSIZE,YHI,VGRAT, VCOLOR, .FALSE.)
              vline = .FALSE.
          ELSE IF (ITYPEX.EQ.1) THEN
	      CALL XAXIS1(XLO,XHI,XTIC,NMX,LINTX,1.,LXB,IFX,XF,YLO,YF,
     *	      XCSIZE,YHI,XDMS,VGRAT, VCOLOR, VLINE, XVALOFF)
	      CALL XAXIS1(XLO,XHI,XTIC,NMX,LINTX,1.,LXB,IFX,XF,YLO,YF,
     *	      XCSIZE,YHI,XDMS,VGRAT, VCOLOR, .FALSE., XVALOFF)
              vline = .FALSE.
          ELSE
	      CALL XAXIS2(XLO,XHI,0.,LXB,IFX,1,XF,YLO,YF,XCSIZE,
     *                    YHI,VGRAT, VCOLOR, VLINE)
	      CALL XAXIS2(XLO,XHI,0.,LXB,IFX,1,XF,YLO,YF,XCSIZE,
     *                    YHI,VGRAT, VCOLOR, .FALSE.)
              vline = .FALSE.
          END IF
      END IF
      enddo  ! ii loop

* TMAP interrupt re-entry
 3333	CONTINUE

	IF(IWIND.GT.0)CALL WINDOW(XLO,YLO,XHI,YHI)
	CALL CROSS
	CALL WINDOW(0.,0.,0.,0.)
C
C	DRAW AXIS LABELS
C
1000	CONTINUE
        IF (mainlab .EQ. 1) CALL LABDRW(XF,YF,IFX,IFY)
	RETURN
	END
