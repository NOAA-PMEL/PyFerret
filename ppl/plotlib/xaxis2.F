      SUBROUTINE XAXIS2(XLO,XHI,XUNIT,LABEL,IFORMT,IAXIS,
     * XFACT,Y,YFACT,CSIZE)
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
C**    @(#)xaxis2.f	1.1    3/10/88
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
C**

	CHARACTER XLAB*20,IFORMT*(*),XFRMT*12
	COMMON/TICS/TSMX,TLX,TSMY,TLY,IX,IY
	COMMON/AXLWID/XWID,YWID
C
C     THIS ROUTINE PLOTS SEVERAL TYPES OF BASE 10 LOG XAXIS.
C         X,YFACT,XFACT HAVE THEIR USUAL MEANINGS.
C         IF CSIZE > 0 THE LABELS HAVE THAT HEIGHT.
C         IF CSIZE < 0 THE LABELS HAVE HEIGHT -CSIZE AND HAVE
C             NEGATIVE THEIR TRUE VALUE.
C         XLO AND XHI ARE THE LOGS OF YOUR XLO AND XHI IN USER UNITS.
C         XUNIT IS ALSO THE LOG OF WHAT IT IS IN USER UNITS.
C         LABEL = -1 FOR LABELING  BELOW OF YOUR X AXIS.
C         LABEL = +1 FOR LABELING ABOVE OF YOUR X AXIS.
C         LABEL =  0 FOR NO LABELS.
C         IFORMT = TO BE FILLED IN LATER.
C
C         IAXIS = STYLE OF AXIS.
C               = 0 FOR NO AXIS.
C               = 1 FOR A SIMPLE LOG AXIS WITH ONE LABEL PER DECADE.
C                  IN THIS CASE XLO, XHI MUST= LOG(POWER OF 10)
C                  AND IFORMT IS IGNORED.
C                  XUNIT MUST = LOG(POWER OF 10)
C               GE 2  THIS GIVES YOU THE INVERSE SCALE OF IAXIS =1
C
      IF(IAXIS.EQ.0) RETURN
      ABSSZ=ABS(CSIZE)
      TXLO=IFIX(1.0001*XLO)
      TXHI=IFIX(1.0001*XHI)
      TXUNIT=IFIX(1.0001*XUNIT)
      NDEC=TXHI-TXLO
!      IF(NDEC.LT.1) STOP 'XLO,XHI'

      IF(NDEC.LT.1) THEN
         NDEC = -1* ndec
         SGN  = TXHI	! use SGN as a dummy
         TXHI = TXLO
         TXLO = SGN
      ENDIF
C
C	DRAW AXIS
C
	IF(IAXIS.EQ.1)THEN
		X=TXLO
		SGN=+1.
	ELSE
		X=TXHI
		SGN=-1.
	ENDIF
	IF  (IAXIS .LE. 3) THEN

	   CALL PLOT(X,Y,0,0)
	   CALL PLOTT7(X,Y,XFACT,YFACT,1,-8)
	   DO 110 I=1,NDEC
	   DO 100 J=2,9
100	   CALL PLOTT7(X+SGN*(I-1+ALOG10(FLOAT(J))),Y,XFACT,YFACT,1,-7)
110	   CALL PLOTT7(X+SGN*I,Y,XFACT,YFACT,1,-8)
	ELSE

C  for DEPTH axes, X plot coordinates are reversed on page.

           XTEMP = X + SGN* NDEC
	   CALL PLOT(XTEMP,Y,0,0)
	   CALL PLOTT7(XTEMP,Y,XFACT,YFACT,1,-8)
	   DO 130 I=1,NDEC
	   DO 120 J=2,9
120	   CALL PLOTT7(XTEMP+(I-1+ALOG10(FLOAT(J))),Y,XFACT,YFACT,1,-7)
130	   CALL PLOTT7(XTEMP+I,Y,XFACT,YFACT,1,-8)

c 120	   CALL PLOTT7(XTEMP+SGN*(I-1+ALOG10(FLOAT(J))),Y,XFACT,YFACT,1,-7)
c 130	   CALL PLOTT7(XTEMP+SGN*I,Y,XFACT,YFACT,1,-8)

        ENDIF
C
C	DRAW LABELS
C
	IF(LABEL.EQ.0)RETURN
	IF(LABEL+IX.EQ.0)THEN
		YTEMP=0.0
	ELSE
		YTEMP=TLX*0.5
	ENDIF
	IF(CSIZE.LT.0.)THEN
		NC=5
		XFRMT='(''-E'',I3)'
	ELSE
		NC=4
		XFRMT='(''E'',I3)'
	ENDIF
	XWID=1.9*ABSSZ+YTEMP
	YTEMP=Y+LABEL*(1.40*ABSSZ+YTEMP)/YFACT
	DO 200 I=0,NDEC

C	   WRITE(XLAB,XFRMT)IFIX(SGN*(X-TXUNIT+I))
	   IF (SGN .GE. 0.) THEN
	      WRITE(XLAB,XFRMT)IFIX(SGN*(X-TXUNIT+SGN*I))
	      XTEMP = X + SGN*I
	   ELSE
	      WRITE(XLAB,XFRMT)IFIX((X-TYUNIT+SGN*(NDEC-I)))
              XTEMP = X + SGN*(NDEC-I)
           ENDIF

	   CALL SCINOT(XTEMP,YTEMP,0.,ABSSZ,NC,XLAB,0,WID,XFACT,YFACT)
200	CONTINUE
	RETURN
	END
