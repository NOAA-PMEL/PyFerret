      SUBROUTINE DAXIS(XF,YF,NMX,NMY,DRAW)

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
C *acm*  5/12  Only draw the axis labels the first time the axes are made.
C              Axes are repeated so axis lines and graticule lines are on top
C              of other plot elements but the labels dont need to be.
C *acm* 10/14  bug 2205. Polygon plots have extra calls because polygons can get 
C              drawn on topof axes. This(?) causes the main label to not get 
C              finished. For polygon plots only, draw the main label twice.
* V694  *acm*  1/15 Changes to allow two time axes on plot (F-T)
* V695 *acm*   4/15 (PyFerret) labels along axes will be drawn with the 
*              tic-mark text style
* V695 *acm*   5/15 Ticket 2279: Draw axis lines, tics and labels only once.
* V7    *acm* 6/16 Ticket 2434: command qualifiers /DEGMINSEC= /HALFSPAC
*                  DEGMINSEC=1 for Deg-Min axis labels, =2 for Deg-Min-Sec. Default = 1
*                  /HALFSPAC adds a half space between the degree sign and E/W or N/S


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
	include 'lines_inc.decl'
	include 'LINES.INC'
	include 'pen_inc.decl'
	include 'PEN.INC'
	include 'pltl_inc.decl'
	include 'PLTL.INC'
	include 'taxis_inc.decl'
	include 'TAXIS.INC'
	include 'tics_inc.decl'
	include 'TICS.INC'
	include 'tics2_inc.decl'
	include 'TICS2.INC'
	include 'axisl_inc.decl'
	include 'AXISL.INC'
	INCLUDE 'pyfonts.cmn'

	CHARACTER IFX*20,IFY*20
	LOGICAL noylab, DRAW

* TMAP interrupt flag
	include 'xinterrupt.cmn'

C If graticule lines are to be drawn across the plot at the tics
C then only do them for one horizontal and one vertical axis.  
C The dash lines interfere with each other if drawn from both 
C sides. HLINE and VLINE control this. Draw the axis after the 
C graticule lines, so the axes themselves are always solid, black.

        HLINE = .TRUE.
        VLINE = .TRUE.

C	SETUP AXIS
	CALL SETLAB(XF,YF,LTT,LTB,LYL,LYR,LXT,LXB,IFX,IFY,LTBb)

C ---------- Skip this section if not drawing the axes ----------
	IF (DRAW) THEN

	noylab =  (ylab .EQ. ' ') 
C
C     DRAW AXIS, LEFT, RIGHT Y AXIS THEN TOP, BOTTOM XAXIS
C
	IF (MAKEP.EQ.'VIEW') GOTO 1000

      
* (PyFerret) labels along vertical axes will be drawn with the ivtics text style
      IF (pyfont .AND. pyf_set(ivtics)) igroup = ivtics

      IF(IAXON(3).EQ.1)THEN
	  IY=ITY
C	  YAXIS1 = LINEAR ,  YAXIS2 = LOG
C
          IF (ITFLG.EQ.-1 .OR. ITFLG.EQ.2) THEN
              IF(ITFLG.EQ.-1) CALL TAYIS(ITYPET,TLO,IT1,IT2,DT,NMTCT,LTB,IFDB,ISDB,
     *        YF,XLO,XF,TCSIZE,XHI,HGRAT, HCOLOR, HLINE, IPEN(0), is_hours_axis)
              IF(ITFLG.EQ.2) CALL TAYIS(ITYPETb,TLOb,IT1b,IT2b,DT,NMTCTb,LTBb,IFDBb,ISDBb,
     *        YF,XLO,XF,TCSIZEb,XHI,HGRAT, HCOLOR, HLINE, IPEN(0), is_hours_axis)
            hline = .FALSE.
	  ELSE IF(ITYPEY.EQ.1)THEN
	   CALL YAXIS1(XLO,XF,YLO,YHI,YTIC,NMY,LINTY,1.,LYL,IFY,YF,
     *	    YCSIZE,XHI,YDMS,YDSPAC,HGRAT, HCOLOR, HLINE, YVALOFF, NOYLAB, 
     *	    SHRINKY, IPEN(0))
            hline = .FALSE.
	  ELSE
            CALL YAXIS2(XLO,XF,YLO,YHI,0.,LYL,IFY,1,YF,YCSIZE,
     .                    XHI,HGRAT, HCOLOR, HLINE, IPEN(0))
            hline = .FALSE.
          END IF
      END IF
	IF ( interrupted ) GOTO 3333		! TMAP interrupt
      IF(IAXON(4).EQ.1)THEN
	IY=-ITY
        IF (ITFLG.EQ.-1 .OR. ITFLG.EQ.2) THEN
            IF(ITFLG.EQ.-1) CALL TAYIS(ITYPET,TLO,IT1,IT2,DT,NMTCT,LTT,IFDB,ISDB,
     *      YF,XHI,XF,TCSIZE,XLO,HGRAT, HCOLOR, HLINE, IPEN(0), is_hours_axis)
            IF(ITFLG.EQ.2) CALL TAYIS(ITYPETb,TLOb,IT1b,IT2b,DT,NMTCTb,LTTb,IFDBb,ISDBb,
     *      YF,XHI,XF,TCSIZEb,XLO,HGRAT, HCOLOR, HLINE, IPEN(0), is_hours_axis)
           hline = .FALSE.
	ELSE IF(ITYPEY.EQ.1)THEN
	    CALL YAXIS1(XHI,XF,YLO,YHI,YTIC,NMY,LINTY,1.,LYR,IFY,YF,
     *     YCSIZE,XLO,YDMS,YDSPAC,HGRAT, HCOLOR, HLINE, YVALOFF, NOYLAB,
     *	    SHRINKY, IPEN(0))
           hline = .FALSE.
	ELSE
	   CALL YAXIS2(XHI,XF,YLO,YHI,0.,LYR,IFY,1,YF,YCSIZE,
     .                    XLO,HGRAT, HCOLOR, HLINE, IPEN(0))
           hline = .FALSE.
	ENDIF
      ENDIF


	IF ( interrupted ) GOTO 3333		! TMAP interrupt

* (PyFerret) labels along horizontal axes will be drawn with the ihtics text style
      IF (pyfont .AND. pyf_set(ihtics)) igroup = ihtics  

      IF(IAXON(1).EQ.1)THEN
          IX=-ITX
C
C	  XAXIS1 = LINEAR, XAXIS2 = LOG, TAXIS CALLS EITHER TAXIS0,
C					 TAXIS1 OR TAXIS3
C
          IF (ITFLG.EQ.1 .OR. ITFLG.EQ.2) THEN
              CALL TAXIS(ITYPET,TLO,IT1,IT2,DT,NMTCT,LTT,IFDB,ISDB,
     *        XF,YHI,YF,TCSIZE,YLO,VGRAT, VCOLOR, VLINE, IPEN(0), is_hours_axis)
              vline = .FALSE.
          ELSE IF (ITYPEX.EQ.1)THEN
	      CALL XAXIS1(XLO,XHI,XTIC,NMX,LINTX,1.,LXT,IFX,XF,YHI,YF,
     *	       XCSIZE,YLO,XDMS,XDSPAC,VGRAT,VCOLOR,VLINE,XVALOFF,IPEN(0))
              vline = .FALSE.
          ELSE
	      CALL XAXIS2(XLO,XHI,0.,LXT,IFX,1,XF,YHI,YF,XCSIZE,
     *                    YLO,VGRAT, VCOLOR, VLINE, IPEN(0))
              vline = .FALSE.
          END IF
      END IF
	IF ( interrupted ) GOTO 3333		! TMAP interrupt
      IF(IAXON(2).EQ.1)THEN
          IX=ITX
          IF (ITFLG.EQ.1 .OR. ITFLG.EQ.2) THEN
              CALL TAXIS(ITYPET,TLO,IT1,IT2,DT,NMTCT,LTB,IFDB,ISDB,
     *        XF,YLO,YF,TCSIZE,YHI,VGRAT, VCOLOR, VLINE, IPEN(0), is_hours_axis)
              vline = .FALSE.
          ELSE IF (ITYPEX.EQ.1) THEN
	      CALL XAXIS1(XLO,XHI,XTIC,NMX,LINTX,1.,LXB,IFX,XF,YLO,YF,
     *	      XCSIZE,YHI,XDMS,XDSPAC,VGRAT, VCOLOR, VLINE, XVALOFF, IPEN(0))
              vline = .FALSE.
          ELSE
	      CALL XAXIS2(XLO,XHI,0.,LXB,IFX,1,XF,YLO,YF,XCSIZE,
     *                    YHI,VGRAT, VCOLOR, VLINE, IPEN(0))
              vline = .FALSE.
          END IF
      END IF

      ENDIF  ! draw
C ---------- End of skip section if not drawing the axes ----------

* TMAP interrupt re-entry
 3333	CONTINUE

	IF(IWIND.GT.0)CALL WINDOW(XLO,YLO,XHI,YHI)
	CALL CROSS
	CALL WINDOW(0.,0.,0.,0.)
C
C	DRAW AXIS LABELS
C
1000	CONTINUE

* (PyFerret) cancel tic-mark text style
	IF (pyfont) igroup = fnt_all

        IF (DRAW) CALL LABDRW(XF,YF,IFX,IFY)
	CALL PPLGFLUSH
	RETURN
	END
