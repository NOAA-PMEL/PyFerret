      SUBROUTINE XAXIS1(XLO,XHI,XTIC,NSMTCX,LINTX,XUNIT,LABEL,
     *    IFORMT,XFACT,Y,YFACT,CSIZE)

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
*	Directory PMEL:[PLOT5.PPLV11.PLOTLIB]XAXIS1.FOR

* search for "TMAP" to find all revisions
* *sh*  - 1/23/89 added interrupt detection
*		  ... no explicit declarations (too much trouble !) so it must
*		      be compiled with /WARNINGS=NODECLARATIONS
*	*jd* 5.6.96 Fix axis labelling bug where roundoff at zero results 
*			in small non-zero values

C**
C**    @(#)xaxis1.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
	CHARACTER IFORMT*(*),XLAB*20
	COMMON/TICS/TSMX,TLX,TSMY,TLY,IX,IY
	COMMON/AXLWID/XWID,YWID

	parameter (epsilon = 1.e-6)


* TMAP interrupt flag
#ifdef unix
	include 'xinterrupt.cmn'
#else
	INCLUDE 'tmap_pplv11inc:tmap_xinterrupt.cmn'
#endif

C     THIS ROUTINE DRAWS AN X AXIS.  IT LABELS THE AXIS BELOW
C     IF LABEL=-1, ABOVE IF LABEL=1, AND NOT AT ALL
C     IF LABEL=0.  IF LABEL.NE.0 IFORMT TELLS US WHAT FORMAT TO WRITE
C     THE LABEL IN.  SEE NCHAR PARAMETER IN COMPLOT DOCUMENTATION
C     FOR THE SUBROUTINE NUMBER FOR AN EXPLANATION OF THIS.
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
CCC
CCC     THE FOLLOWING SECTION OF CODE MAKES THIS ROUTINE UPWARD COMPATIABLE.
        ABSSZ=ABS(CSIZE)
        SIGSZ=SIGN(1.,CSIZE)
C
C	DRAW AXIS
C
	DIR=SIGN(1.,XTIC)
	XT=INT(XLO/XTIC+SIGN(0.000001,DIR*XLO))*XTIC
	IF(DIR*XT.LT.DIR*XLO)XT=XT+XTIC
	ISTOP=IFIX((XHI-XT)/XTIC+0.000001)
	CALL PLOT(XT,Y,0,0)
	CALL SMX1(XT,XLO,-XTIC,NSMTCX,XFACT,YFACT,Y)
	CALL PLOT(XLO,Y,1,0)
	CALL PLOTT7(XT,Y,XFACT,YFACT,1,-8)
	X=XT
	DO 100 I=1,ISTOP
	IF ( interrupted ) RETURN		! TMAP interrupt
	CALL SMX1(X,XHI,XTIC,NSMTCX,XFACT,YFACT,Y)
	X=X+XTIC
	CALL PLOTT7(X,Y,XFACT,YFACT,1,-8)
100	CONTINUE
	CALL SMX1(X,XHI,XTIC,NSMTCX,XFACT,YFACT,Y)
	CALL PLOT(XHI,Y,1,0)
C
C	DRAW LABELS
C
	IF(LABEL.EQ.0)RETURN
	IF(LABEL+IX.EQ.0)THEN
		YTEMP=0.0
	ELSE
		YTEMP=TLX*0.5
	ENDIF
	XWID=1.9*ABSSZ+YTEMP
	YTEMP=Y+LABEL*(1.40*ABSSZ+YTEMP)/YFACT
	IF(DIR*XLO.LE.0.AND.DIR*XHI.GE.0)THEN
		X=INT(XLO/(XTIC*LINTX)-0.000001)*XTIC*LINTX
	ELSE
		X=XT
	ENDIF
	ISTOP=INT((XHI-X)/(XTIC*LINTX)+0.000001)

	x0 = x
	DO 200 I=0,ISTOP
	IF ( interrupted ) RETURN		! TMAP interrupt
        TEMP=SIGSZ*X/XUNIT
        CALL NUMCDE(TEMP,XLAB,NC,IFORMT)
        CALL SCINOT(X,YTEMP,0.,ABSSZ,NC,XLAB,0,WID,XFACT,YFACT)

CC	Original code
CC	X=X+XTIC*LINTX

	X=x0+XTIC*LINTX*(i+1)
	if (abs(x) .lt. epsilon*abs(x0)) x = 0.0

200	CONTINUE
	RETURN
	END
