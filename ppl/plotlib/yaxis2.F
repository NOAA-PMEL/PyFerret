      SUBROUTINE YAXIS2(X,XFACT,YLO,YHI,YUNIT,LABEL,IFORMT,IAXIS,
     * YFACT,CSIZE,xacross,hgrat,hcolor,hline)
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
C**    @(#)yaxis2.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
* *acm* - 6/01   changes to allow plot qualifiers to set log axes,
*                inverse log axes and log depth axes.

	CHARACTER YLAB*20,IFORMT*(*),YFRMT*12
	COMMON/TICS/TSMX,TLX,TSMY,TLY,IX,IY
	COMMON/AXLWID/XWID,YWID

* Parameters for drawing tics across plot

        REAL xacross
        REAL dash1_1, dash1_2, dash1_3, dash1_4 
        REAL dash2_1, dash2_2, dash2_3, dash2_4
        REAL hgrat(4,2)        ! line type: 0=tics only, <0=line, >0=dash lengths 
        REAL yt
        INTEGER hcolor(*)      ! pen number 
        LOGICAL hline          ! If true, do the lines across.
        LOGICAL solid1,solid2  ! solid lines across at large, small tics
        LOGICAL dashed1, dashed2 ! dashed lines across at large, small tics

C
C     THIS ROUTINE PLOTS SEVERAL TYPES OF BASE 10 LOG YAXIS.
C         X,XFACT,YFACT HAVE THEIR USUAL MEANINGS.
C         IF CSIZE > 0 THE LABELS HAVE THAT HEIGHT.
C         IF CSIZE < 0 THE LABELS HAVE HEIGHT -CSIZE AND HAVE
C             NEGATIVE THEIR TRUE VALUE.
C         YLO AND YHI ARE THE LOGS OF YOUR YLO AND YHI IN USER UNITS.
C         YUNIT IS ALSO THE LOG OF WHAT IT IS IN USER UNITS.
C         LABEL = -1 FOR LABELING ON THE LEFT OF YOUR Y AXIS.
C         LABEL = +1 FOR LABELING ON THE RIGHT OF YOUR Y AXIS.
C         LABEL =  0 FOR NO LABELS.
C         IFORMT = TO BE FILLED IN LATER.
C
C         IAXIS = STYLE OF AXIS.
C                = 0 FOR NO AXIS
C                = 1 FOR A SIMPLE LOG AXIS WITH ONE LABEL PER DECADE.
C                  IN THIS CASE YLO, YHI MUST= LOG(POWER OF 10)
C                  AND IFORMT IS IGNORED.
C                  YUNIT MUST = LOG(POWER OF 10)
C               GE 2  GIVES THE INVERSE SCALE OF IAXIS = 1

      IF(IAXIS.EQ.0) RETURN
      ABSSZ=ABS(CSIZE)
      TYLO=IFIX(1.0001*YLO)
      TYHI=IFIX(1.0001*YHI)
      TYUNIT=IFIX(1.0001*YUNIT)
      NDEC=TYHI-TYLO
!      IF(NDEC.LT.1) STOP 'YLO,YHI'
      IF(NDEC.LT.1) THEN
         NDEC = -1* ndec
         SGN  = TYHI	! use SGN as a dummy
         TYHI = TYLO
         TYLO = SGN
      ENDIF

C
C	DRAW AXIS
C
	IF(IAXIS.EQ.1)THEN
		Y=TYLO
		SGN= 1.0
	ELSE
		Y=TYHI
		SGN=-1.0
	ENDIF
 
        solid1 = (hgrat(1,1).LT.0 .AND. hgrat(2,1).LT.0 .AND.  
     .            hgrat(3,1).LT.0 .AND.  hgrat(4,1).LT.0 )
        solid2 = (hgrat(1,2).LT.0 .AND. hgrat(2,2).LT.0 .AND.  
     .            hgrat(3,2).LT.0 .AND.  hgrat(4,2).LT.0 )
        dashed1 = (hgrat(1,1).GT.0 .OR. hgrat(2,1).GT.0 .OR.  
     .            hgrat(3,1).GT.0 .OR.  hgrat(4,1).GT.0 )
        dashed2 = (hgrat(1,2).GT.0 .OR. hgrat(2,2).GT.0 .OR.  
     .            hgrat(3,2).GT.0 .OR.  hgrat(4,2).GT.0 )
        IF (dashed1) THEN
           dash1_1 = hgrat(1,1)
           dash1_2 = hgrat(2,1)
           dash1_3 = hgrat(3,1)
           dash1_4 = hgrat(4,1)
        ENDIF
        IF (dashed2) THEN
           dash2_1 = hgrat(1,2)
           dash2_2 = hgrat(2,2)
           dash2_3 = hgrat(3,2)
           dash2_4 = hgrat(4,2)
        ENDIF

	IF  (IAXIS .LE. 3) THEN
	   CALL PLOT(X,Y,0,0)
	   CALL PLOTT7(X,Y,XFACT,YFACT,1,-6)
	   DO 110 I=1,NDEC    ! large tics

	      DO 100 J=2,9    ! small tics

                 CALL PLOTT7(X,Y+SGN*(I-1+ALOG10(FLOAT(J))),XFACT,YFACT,1,-5)

                 IF ((solid2.OR.dashed2) .AND. hline) THEN ! make line across
                   yt = Y+SGN*(I-1+ALOG10(FLOAT(J)))
                   CALL COLOR(hcolor(2))
                    IF (dashed2) THEN
                       CALL DASHSZ(dash2_1,dash2_2,dash2_3,dash2_4)
                       CALL DASHES
                    ENDIF
                    CALL PLOT(X,yt,0,0) 
                    CALL PLOT(xacross,yt,1,0)
                    CALL PLOT(X,yt,0,0) 
                    CALL COLOR(1)
                    CALL VECTRS
                 ENDIF
             
100           CONTINUE

              CALL PLOTT7(X,Y+SGN*I,XFACT,YFACT,1,-6) ! draw large tic

              IF ((solid1.OR.dashed1) .AND. hline) THEN ! make line across
                 yt = Y+SGN*I
                 CALL COLOR(hcolor(1))
                 IF (dashed1) THEN
                    CALL DASHSZ(dash1_1,dash1_2,dash1_3,dash1_4)
                    CALL DASHES
                 ENDIF
                 CALL PLOT(X,yt,0,0) 
                 CALL PLOT(xacross,yt,1,0)
                 CALL PLOT(X,yt,0,0) 
                 CALL COLOR(1)
                 CALL VECTRS
              ENDIF

110        CONTINUE

	ELSE

C  for DEPTH axes, Y plot coordinates are reversed on page.

           YTEMP = Y + SGN* NDEC
	   CALL PLOT(X,YTEMP,0,0)
	   CALL PLOTT7(X,YTEMP,XFACT,YFACT,1,-6)
	   DO 130 I=1,NDEC      ! large tics
	      DO 120 J=2,9      ! small tics
                 CALL PLOTT7(X,YTEMP+(I-1+ALOG10(FLOAT(J))),XFACT,YFACT,1,-5)

                 IF ((solid2.OR.dashed2) .AND. hline) THEN ! make line across
                   yt = YTEMP+(I-1+ALOG10(FLOAT(J)))
                   CALL COLOR(hcolor(2))
                    IF (dashed2) THEN
                       CALL DASHSZ(dash2_1,dash2_2,dash2_3,dash2_4)
                       CALL DASHES
                    ENDIF
                    CALL PLOT(X,yt,0,0) 
                    CALL PLOT(xacross,yt,1,0)
                    CALL PLOT(X,yt,0,0) 
                    CALL COLOR(1)
                    CALL VECTRS
                 ENDIF

120           CONTINUE
              CALL PLOTT7(X,YTEMP+I,XFACT,YFACT,1,-6)  ! make large tic
              
              IF ((solid1.OR.dashed1) .AND. hline) THEN ! make line across
                 yt = ytemp+i
                 CALL COLOR(hcolor(1))
                 IF (dashed1) THEN
                    CALL DASHSZ(dash1_1,dash1_2,dash1_3,dash1_4)
                    CALL DASHES
                 ENDIF
                 CALL PLOT(X,yt,0,0) 
                 CALL PLOT(xacross,yt,1,0)
                 CALL PLOT(X,yt3/2/2004,0,0) 
                 CALL COLOR(1)
                 CALL VECTRS
              ENDIF

130        CONTINUE
        ENDIF
C
C	DRAW LABELS
C
	IF(LABEL.EQ.0)RETURN
	IF(LABEL+IY.EQ.0)THEN
		XTEMP=0.0
	ELSE
		XTEMP=TLY*0.5
	ENDIF
	IF(LABEL.EQ.1)THEN
		IPOS=2
	ELSE
		IPOS=1
	ENDIF
	YWID=0.75*ABSSZ+XTEMP
	XTEMP=X+LABEL*YWID/XFACT
	IF(CSIZE.LT.0.)THEN
		NC=5
		YFRMT='(''-E'',I3)'
	ELSE
		NC=4
		YFRMT='(''E'',I3)'
	ENDIF
	YWIDT=0.0
	DO 200 I=0,NDEC
c	WRITE(YLAB,YFRMT)IFIX(SGN*(Y-TYUNIT+I))	! wrong nums for log depth axis 

	IF (SGN .GE. 0) THEN
            WRITE(YLAB,YFRMT)IFIX((Y-TYUNIT+SGN*I))
            YTEMP = Y + SGN*I

C  for DEPTH axes, Y plot coordinates are reversed on page.

        ELSE IF (SGN .LT. 0.)THEN
            WRITE(YLAB,YFRMT)IFIX((Y-TYUNIT+SGN*(NDEC-I)))
            YTEMP = Y + SGN*(NDEC-I)
        ENDIF

	CALL SCINOT(XTEMP,YTEMP,0.,ABSSZ,NC,YLAB,IPOS,WID,XFACT,YFACT)

	IF(WID.GT.YWIDT)YWIDT=WID
200	CONTINUE
	YWID=YWID+YWIDT
	RETURN
	END
