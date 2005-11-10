      SUBROUTINE YAXIS1(X,XFACT,YLO,YHI,YTIC,NSMTCY,LINTY,YUNIT,LABEL,
     *    IFORMT,YFACT,CSIZE,xacross,ydms,hgrat,hcolor,hline)

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
        INTEGER ydms
* Parameters for drawing tics across plot

        REAL xacross, dash1, dash2, dash3, dash4
        REAL hgrat(4,2)        ! line type: 0=tics only, <0=line, >0=dash lengths 
        INTEGER hcolor(*)      ! pen number 
        LOGICAL hline          ! If true, do the lines across.
        LOGICAL solid          ! solid lines across at tics
        LOGICAL dashed         ! dashed lines across at tics

	CHARACTER IFORMT*(*),YLAB*20
	COMMON/TICS/TSMX,TLX,TSMY,TLY,IX,IY
	COMMON/AXLWID/XWID,YWID 

	parameter (epsilon = 1.e-6)

* TMAP interrupt flag
#ifdef unix
	include 'xinterrupt.cmn'
#else
	INCLUDE 'tmap_pplv11inc:tmap_xinterrupt.cmn'
#endif
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
	CALL PLOT(X,YLO,1,0)
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
        IF ((solid.OR.dashed) .AND. hline) THEN ! make line across
           CALL COLOR(hcolor(1))
           IF (dashed) THEN
              CALL DASHSZ(dash1,dash2,dash3,dash4)
              CALL DASHES
           ENDIF
           CALL PLOT(X,Y,0,0) 
           CALL PLOT(xacross,Y,1,0)
           CALL PLOT(X,Y,0,0) 
           CALL COLOR(1)
           CALL VECTRS
        ENDIF

	Y=Y+YTIC
	CALL PLOTT7(X,Y,XFACT,YFACT,1,-6)
100	CONTINUE
	CALL SMY1(X,XFACT,YFACT,Y,YHI,YTIC,NSMTCY,
     .     xacross,hgrat,hcolor,hline)
        
        IF ((solid.OR.dashed) .AND. hline) THEN ! make line across
           CALL COLOR(hcolor(1))
           IF (dashed) THEN
              CALL DASHSZ(dash1,dash2,dash3,dash4)
              CALL DASHES
           ENDIF
           CALL PLOT(X,Y,0,0) 
           CALL PLOT(xacross,Y,1,0)
           CALL PLOT(X,Y,0,0) 
           CALL COLOR(1)
           CALL VECTRS
        ENDIF

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
	IF(LABEL.EQ.1)THEN
		CALL NUMCDE(SIGSZ*YLO/YUNIT,YLAB,NC,IFORMT,ydms)
		CALL SCINOT(0.,0.,0.,ABSSZ,NC,YLAB,3,WID,
     .             XFACT,YFACT)
		CALL NUMCDE(SIGSZ*YHI/YUNIT,YLAB,NC,IFORMT,ydms)
		CALL SCINOT(0.,0.,0.,ABSSZ,NC,YLAB,3,WID1,
     .             XFACT,YFACT)
		IF(YLO.LT.0..AND.YHI.GT.0.)THEN
			CALL NUMCDE(0.,YLAB,NC,IFORMT,ydms)
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
	YWIDT=0.

	y0 = y
	DO 200 I=0,ISTOP
	IF ( interrupted ) RETURN		! TMAP interrupt
        TEMP=SIGSZ*Y/YUNIT
        CALL NUMCDE(TEMP,YLAB,NC,IFORMT,YDMS)
        CALL SCINOT(XTEMP,Y,0.,ABSSZ,NC,YLAB,1,WID,XFACT,YFACT)

CC	Original code
CC	Y=Y+YTIC*LINTY

	Y=y0+YTIC*LINTY*(i+1)
	if (abs(y) .lt. epsilon*abs(y0)) y = 0.0

	IF(WID.GT.YWIDT)YWIDT=WID
200	CONTINUE
	IF(LABEL.LT.0)THEN 
           IF (YDMS .GE. 2) ywidt = ywidt - symwid(abssz,1,'''')
           YWID=YWID+YWIDT
        ENDIF
	RETURN
	END


