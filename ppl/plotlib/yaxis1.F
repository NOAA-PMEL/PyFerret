      SUBROUTINE YAXIS1(X,XFACT,YLO,YHI,YTIC,NSMTCY,LINTY,YUNIT,LABEL,
     *    IFORMT,YFACT,CSIZE,xacross,ydms,ydspac,hgrat,hcolor,hline,
     *    yvaloff, noylab, SHRINKY, color0)

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
*	Directory PMEL:[PLOT5.PPLV11.PLOTLIB]YAXIS1.FOR

* search for "TMAP" to find all revisions
* *sh*  - 1/23/89 added interrupt detection
*		  ... no explicit declarations (too much trouble !) so it must
*		      be compiled with /WARNINGS=NODECLARATIONS
*	*jd* 5.6.96 Fix axis labelling bug where roundoff at zero results 
*			in small non-zero values
* v554: *acm* 2/04 add /HGRAT /VGRAT to draw optional graticule lines at tic marks
C *acm* 11/05  new argument to scinot deg/min/sec labels
C        1/06  xvaloff,yvaloff add this amount to each x or y axis label. 
C              (axis data was read as double precision and converted to single 
C              after applying an offset.)

* 11/2011 ACM Fixing ticket 1896: When needed, reset the size of numerical axis
*             labes, so the labels dont run off the edge of the plot.
* V7    *acm* 6/16 Ticket 2434: command qualifiers /DEGMINSEC= /HALFSPAC
*                  DEGMINSEC=1 for Deg-Min axis labels, =2 for Deg-Min-Sec. Default = 1
*                  /HALFSPAC adds a half space between the degree sign and E/W or N/S
* *acm* - 8/16   ticket 1787: restore pen 0 correctly after graticule
* *acm* - 11/16  ticket 2477: incomplete axis on some plots
* V745 *acm* 3/19 Draw tic marks on top of any graticule lines so 
*                 dashes or color are covered up by the tic mark
* V766 *acm*11/22 Fix issue 129: 1st graticule line was not drawn

C**
C**    @(#)yaxis1.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
        INTEGER ydms, ydspac
	CHARACTER*1 letter
	REAL SYMWID, spacwid
	LOGICAL do_shift

* Parameters for drawing tics across plot

        REAL xacross, dash1, dash2, dash3, dash4
        REAL hgrat(4,2)        ! line type: 0=tics only, <0=line, >0=dash lengths 
        INTEGER hcolor(*)      ! pen number for graticule lines
        INTEGER color0         ! pen number to restore
	LOGICAL hline          ! If true, do the lines across.
        LOGICAL solid          ! solid lines across at tics
        LOGICAL dashed         ! dashed lines across at tics
	LOGICAL noylab
	LOGICAL SHRINKY

	CHARACTER IFORMT*(*),YLAB*20
	COMMON/TICS/TSMX,TLX,TSMY,TLY,IX,IY
	COMMON/AXLWID/XWID,YWID 

	parameter (epsilon = 1.e-6)

* TMAP interrupt flag
	include 'xinterrupt.cmn'
        include 'parampl5_dat.decl'
        include 'PARAMPL5.DAT'
        include 'lines_inc.decl'
        include 'LINES.INC'
        INCLUDE 'pen_inc.decl'  ! with IPEN(1)
        INCLUDE 'PEN.INC'
C
C
C     THIS ROUTINE DRAWS A Y AXIS.  IT LABELS THE AXIS ON THE LEFT
C     IF LABEL=-1, ON THE RIGHT IF LABEL=1, AND NOT AT ALL
C     IF LABEL=0.  IF LABEL.NE.0 IFORMT TELLS US WHAT FORMAT TO WRITE
C     THE LABEL IN.  SEE NCHAR PARAMETER IN COMPLOT DOCUMENTATION
C     FOR THE SUBROUTINE NUMBER FOR AN EXPLANATION OF THIS.
C     IF CSIZE>0 THE LABELS HAVE THAT SIZE.  IF CSIZE<0 THE LABELS
C     HAVE THE SIZE -CSIZE  AND HAVE THE NEGATIVE OF THEIR TRUE
C     VALUES.
C     MODIFICATION OF JULY 29,1979.
C     WE HAVE CHANGED THE MEANING OF IFORMT.
C     USE -10<=IFORMT<=+10 AS BEFORE OR BETTER YET:
C     USE IFORMT = 10H(I7*BALL*)
C                 = 7H(F10.2)  FOR EXAMPLE
C     IN THIS OPTION OUR SUBROUTINE RECONIZES AN I FORMAT BY USING
C     THE FACT THAT THE FIRST TWO CHARACTERS OF IFORMT ARE "(I".
C     IN FACT IF YOU DIMENSION IFORMT IN THE CALLING PROGRAM YOU
C     CAN EVEN USE FORMAT STATEMENTS THAT ARE LONGER THAN 10 CHARACTERS.
C     HOWEVER, THE PRINTED LABEL MUST BE <=20 CHARACTERS.
C
C     THE FOLLOWING SECTION OF CODE MAKES THIS ROUTINE UPWARD COMPATIABLE.
      ABSSZ=ABS(CSIZE)
      SIGSZ=SIGN(1.,CSIZE)
C
C	DRAW AXIS
C

	DIR=SIGN(1.,YTIC)
	YT=INT(YLO/YTIC+SIGN(0.000001,DIR*YLO))*YTIC
	IF(DIR*YT.LT.DIR*YLO)YT=YT+YTIC
	ISTOP=IFIX((YHI-YT)/YTIC+0.000001)
	CALL PLOT(X,YT,0,0)
	CALL SMY1(X,XFACT,YFACT,YT,YLO,-YTIC,NSMTCY,
     .     xacross,hgrat,hcolor,hline)

	CALL PLOTT7(X,YT,XFACT,YFACT,1,-6)

	Y=YT
        
        solid = (hgrat(1,1).LT.0 .AND. hgrat(2,1).LT.0 .AND.  
     .            hgrat(3,1).LT.0 .AND.  hgrat(4,1).LT.0 )
        dashed = (hgrat(1,1).GT.0 .OR. hgrat(2,1).GT.0 .OR.  
     .            hgrat(3,1).GT.0 .OR.  hgrat(4,1).GT.0 )
        IF (dashed) THEN
           dash1 = hgrat(1,1)
           dash2 = hgrat(2,1)
           dash3 = hgrat(3,1)
           dash4 = hgrat(4,1)
        ENDIF

	DO 100 I=1,ISTOP
	IF ( interrupted ) RETURN		! TMAP interrupt
	CALL SMY1(X,XFACT,YFACT,Y,YHI,YTIC,NSMTCY,
     .     xacross,hgrat,hcolor,hline)
        IF ((solid.OR.dashed) .AND. hline .AND. y.GT.ylo) THEN ! make line across
           CALL COLOR(hcolor(1))
           IF (dashed) THEN
              CALL DASHSZ(dash1,dash2,dash3,dash4)
              CALL DASHES
           ENDIF
           CALL PLOT(X,Y,0,0)
           CALL PLOT(xacross,Y,1,0)
           CALL PLOT(X,Y,0,0) 
           CALL COLOR(color0)
           CALL VECTRS
        ENDIF
	CALL PLOTT7(X,Y,XFACT,YFACT,1,-6)

	Y=Y+YTIC
100	CONTINUE
	CALL SMY1(X,XFACT,YFACT,Y,YHI,YTIC,NSMTCY,
     .     xacross,hgrat,hcolor,hline)
        
        IF ((solid.OR.dashed) .AND. hline .AND. y.LT.yhi) THEN ! make line across
           CALL COLOR(hcolor(1))
           IF (dashed) THEN
              CALL DASHSZ(dash1,dash2,dash3,dash4)
              CALL DASHES
           ENDIF
           CALL PLOT(X,Y,0,0) 
           CALL PLOT(xacross,Y,1,0)
           CALL PLOT(X,Y,0,0) 
           CALL COLOR(color0)
           CALL VECTRS
        ENDIF
	
	CALL PLOTT7(X,Y,XFACT,YFACT,1,-6)

	CALL PLOT(X,YLO,1,0)
	CALL PLOT(X,YHI,1,0)
C
C	DRAW LABELS
C
	IF(LABEL.EQ.0)RETURN
	IF(LABEL+IY.EQ.0)THEN
		XTEMP=0.
	ELSE
		XTEMP=TLY*0.5
	ENDIF	
	
	LETTER = ' '
	IF(LABEL.EQ.1)THEN
           IF (yvaloff .EQ. 0.) THEN
	      CALL NUMCDE(SIGSZ*YLO/YUNIT,YLAB,LETTER,NC,IFORMT,ydms)
           ELSE
	      CALL NUMCDE_OFF(SIGSZ*YLO/YUNIT,YLAB,LETTER,NC,IFORMT,ydms,
     .              yvaloff)
           ENDIF
	   CALL SCINOT(0.,0.,0.,ABSSZ,NC,YLAB,3,WID,
     .             XFACT,YFACT)

           IF (yvaloff .EQ. 0.) THEN
	      CALL NUMCDE(SIGSZ*YHI/YUNIT,YLAB,LETTER,NC,IFORMT,ydms)
           ELSE
	      CALL NUMCDE_OFF(SIGSZ*YHI/YUNIT,YLAB,LETTER,NC,IFORMT,ydms,yvaloff)
           ENDIF
	   CALL SCINOT(0.,0.,0.,ABSSZ,NC,YLAB,3,WID1,
     .             XFACT,YFACT)

	   IF(YLO.LT.0..AND.YHI.GT.0.)THEN
              IF (yvaloff .EQ. 0.) THEN
                 CALL NUMCDE(0.,YLAB,LETTER,NC,IFORMT,ydms)
              ELSE
                 CALL NUMCDE_OFF(0.,YLAB,LETTER,NC,IFORMT,ydms,yvaloff)
              ENDIF
	      CALL SCINOT(0.,0.,0.,ABSSZ,NC,YLAB,3,WID2,
     *			XFACT,YFACT)
	   ELSE
	      WID2=0.0
	   ENDIF
	   XTEMP=XTEMP+AMAX1(WID,WID1,WID2)
	ENDIF
	
	YWID=0.75*ABSSZ+XTEMP
	XTEMP=X+LABEL*YWID/XFACT  

	IF(DIR*YLO.LE.0.AND.DIR*YHI.GE.0)THEN
		Y=INT(YLO/(YTIC*LINTY)-0.000001)*YTIC*LINTY
	ELSE
		Y=YT
	ENDIF
	ISTOP=INT((YHI-Y)/(YTIC*LINTY)+0.000001)

	y0 = y
	YWIDT=0.
        YSAVE = Y

	IF (SHRINKY) THEN
C First compute max axis-label width without writing labels 
C (ITYPE=3 arg to SCINOT) 

        DO 199 I=0,ISTOP
	IF ( interrupted ) RETURN		! TMAP interrupt
        TEMP=SIGSZ*Y/YUNIT
        IF (yvaloff .EQ. 0) THEN
           CALL NUMCDE(TEMP,YLAB,LETTER,NC,IFORMT,ydms)
        ELSE
           CALL NUMCDE_OFF(TEMP,YLAB,LETTER,NC,IFORMT,ydms,yvaloff)
        ENDIF
        CALL SCINOT(XTEMP,Y,0.,ABSSZ,NC,YLAB,3,WID,XFACT,YFACT)
	
	Y=y0+YTIC*LINTY*(i+1)
	if (abs(y) .lt. epsilon*abs(y0)) y = 0.0

	IF(WID.GT.YWIDT)YWIDT=WID
199	CONTINUE

* Use origin and label-size info that is in common to possibly
* reset the axis-label size for this axis only, so the labels dont
* run off the edge. (Only do this if the axis is labeled on the left.
* means we are ignoring axes at the right of the page, labeled on the right.)

        IF (label .EQ. -1) CALL RESET_LABSIZ (YWID, YWIDT, ABSSZ, NOYLAB)

	ENDIF !SHRINKY

	YWIDT=0.
	Y = YSAVE
	LETTER = ' '

        TEMP=SIGSZ*Y/YUNIT
	CALL NUMCDE(TEMP,YLAB,LETTER,NC,IFORMT,ydms)
	
	spacwid = 0.
	do_shift = ( INDEX(YLAB, '#') .GT. 0 )  ! longitude or latitude labels
	IF (do_shift) spacwid = SYMWID(ABSSZ,1,'N')

* Now make the labels. 
	DO 200 I=0,ISTOP
	IF ( interrupted ) RETURN		! TMAP interrupt
        TEMP=SIGSZ*Y/YUNIT
        IF (yvaloff .EQ. 0) THEN
           CALL NUMCDE(TEMP,YLAB,LETTER,NC,IFORMT,ydms)
        ELSE
           CALL NUMCDE_OFF(TEMP,YLAB,LETTER,NC,IFORMT,ydms,yvaloff)
        ENDIF

* Letter E/W or N/S is added separatly.  Need a space for that plus
* any extra requested space.

	IF (do_shift) spacwid = SYMWID(ABSSZ,1,LETTER)

	XL = XTEMP
	IF (label .EQ. -1) THEN
	   IF (ydspac .GT. 0) THEN
	      XL = XTEMP - (0.5*spacwid + 0.5*ydspac*spacwid)/XFACT 
	   ELSE
	      XL = XTEMP - spacwid/XFACT 
	   ENDIF
	ENDIF

        CALL SCINOT(XL,Y,0.,ABSSZ,NC,YLAB,1,WID,XFACT,YFACT)
        IF (letter .NE. ' ') THEN
	   NL = 1
	   XL = XL + 0.5*(0.5*ydspac*spacwid)/XFACT  ! SCINOT called with right-justify
	   YL = Y - 0.5*ABSSZ/YFACT             ! SCINOT centers vertically
	   CALL SYMBEL(XL,YL,0.,ABSSZ,NC,LETTER)
	ENDIF

CC	Original code
CC	Y=Y+YTIC*LINTY

	Y=y0+YTIC*LINTY*(i+1)
	if (abs(y) .lt. epsilon*abs(y0)) y = 0.0

	IF(WID.GT.YWIDT) YWIDT=WID 
200	CONTINUE
	IF(LABEL.LT.0)THEN 
           IF (YDMS .GE. 2) ywidt = ywidt - symwid(abssz,1,'''')
           YWID=YWID+YWIDT
        ENDIF
	
	YWID = YWID + spacwid + 0.5*ydspac*spacwid
	RETURN
	END


