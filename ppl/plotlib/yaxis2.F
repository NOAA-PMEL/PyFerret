      SUBROUTINE YAXIS2(X,XFACT,YLO,YHI,YUNIT,LABEL,IFORMT,IAXIS,
     * YFACT,CSIZE)
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
	CHARACTER YLAB*20,IFORMT*(*),YFRMT*12
	COMMON/TICS/TSMX,TLX,TSMY,TLY,IX,IY
	COMMON/AXLWID/XWID,YWID
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
C                = 2   THIS GIVES YOU THE INVERSE SCALE OF IAXIS = 1
C
      IF(IAXIS.EQ.0) RETURN
      ABSSZ=ABS(CSIZE)
      TYLO=IFIX(1.0001*YLO)
      TYHI=IFIX(1.0001*YHI)
      TYUNIT=IFIX(1.0001*YUNIT)
      NDEC=TYHI-TYLO
      IF(NDEC.LT.1) STOP 'YLO,YHI'
C
C	DRAW AXIS
C
	IF(IAXIS.EQ.1)THEN
		Y=TYLO
		SGN=+1.0
	ELSE
		Y=TYHI
		SGN=-1.0
	ENDIF
	CALL PLOT(X,Y,0,0)
	CALL PLOTT7(X,Y,XFACT,YFACT,1,-6)
	DO 100 I=1,NDEC
	DO 150 J=2,9
150	CALL PLOTT7(X,Y+SGN*(I-1+ALOG10(FLOAT(J))),XFACT,YFACT,1,-5)
100	CALL PLOTT7(X,Y+SGN*I,XFACT,YFACT,1,-6)
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
	WRITE(YLAB,YFRMT)IFIX(SGN*(Y-TYUNIT+I))
	CALL SCINOT(XTEMP,Y+SGN*I,0.,ABSSZ,NC,YLAB,IPOS,WID,XFACT,YFACT)
	IF(WID.GT.YWIDT)YWIDT=WID
200	CONTINUE
	YWID=YWID+YWIDT
	RETURN
	END
