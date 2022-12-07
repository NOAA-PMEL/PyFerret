	SUBROUTINE LINEAR(Z,NXSIZE,NYSIZE,NX,NY,X1,Y1,DX,DY,
     *	XP,YP,ZP,N,CAY,NRNG)
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
* 11/2022 *acm* Code cleanup: Use the variable from common CMXGE rather than 1.e35
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
*
C**    @(#)linear.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C
C	This is a substitute routine for ZGRID, where the interpolation
C	used is linear rather than a fancy spline or lagrangian fit.
C	While the technique is crude the results are such that no
C	spurious minimums or maximums will be created.  This routine
C	expects the input data (XP,YP,ZP) to be ordered and complete.
C
C	The data may not have any interal gaps, but 1.0e35 is a valid
C	missing data value.  The only exception to the data being
C	complete is that the variable that varies fastest may end 
C	early, however, the data must start together.
C
C	This subroutine expects the data to be on a grid for which the
C	rows and columns can be unevenly spaced.  If the data is by
C	rows then a row may be short or if by columns a column can be
C	short.  (any other gaps will produce unpredictable results)
C
C	programmed by: D. W. Denbo    Mar 1986
C	modified by: D. W. Denbo   July 1986
C		added missing data types, and speeded it up
C
C
	INCLUDE 'MISS.INC'
C
	DIMENSION Z(NXSIZE,NYSIZE),XP(N),YP(N),ZP(N)
	DIMENSION X(20000),Y(20000),IPNT(20000),IEND(20000)
	DIMENSION IPS(20000),ODX(20000),ODY(20000)
	LOGICAL XDIR,YDIR,XIN,YIN,XFAST,FLIP
	INTEGER I,J,K,L
C
C	Set up for missing data
C
	BIG=0.9E35
	ZP(N+1)=1.0E35
C
C	Scan the input series XP,YP to determine the structure
C
	NIX=1
	NIY=1
	X(1)=XP(1)
	Y(1)=YP(1)
	IPNT(1)=1
C
	XFAST=XP(1).NE.XP(2)
	IF(XFAST)THEN
	    DO 10 I=2,N
	    IF(XP(I).EQ.XP(1))GOTO 11
	    NIX=NIX+1
	    X(NIX)=XP(I)
10	    CONTINUE
11	    DO 12 I=1,N
C
C	TEST FOR MISSING DATA
C
	    IF(ZLE.AND.ZP(I).LE.CMZLE)ZP(I)=cmxge
	    IF(ZEQ.AND.ZP(I).EQ.CMZEQ)ZP(I)=cmxge
	    IF(ZGE.AND.ZP(I).GE.CMZGE)ZP(I)=cmxge
	    IF(YP(I).NE.Y(NIY))THEN
		NIY=NIY+1
		Y(NIY)=YP(I)
		IPNT(NIY)=I
	    ENDIF
12	    CONTINUE
	    IPNT(NIY+1)=N+1
	ELSE
	    DO 13 I=2,N
	    IF(YP(I).EQ.YP(1))GOTO 14
	    NIY=NIY+1
	    Y(NIY)=YP(I)
13	    CONTINUE
14	    DO 15 I=1,N
C
C	TEST FOR MISSING DATA
C
	    IF(ZLE.AND.ZP(I).LE.CMZLE)ZP(I)=cmxge
	    IF(ZEQ.AND.ZP(I).EQ.CMZEQ)ZP(I)=cmxge
	    IF(ZGE.AND.ZP(I).GE.CMZGE)ZP(I)=cmxge
	    IF(XP(I).NE.X(NIX))THEN
		NIX=NIX+1
		X(NIX)=XP(I)
		IPNT(NIX)=I
	    ENDIF
15	    CONTINUE
	    IPNT(NIX+1)=N+1
	ENDIF
C
	IF(XFAST)THEN
	    DO 420 I=1,NIY
	      IEND(I)=IPNT(I+1)
420	    CONTINUE
	ELSE
	    DO 421I=1,NIX
	      IEND(I)=IPNT(I+1)
421	    CONTINUE
	ENDIF
C
	XDIR=X(2).GT.X(1)
	YDIR=Y(2).GT.Y(1)
C
C	Check for the beginning of the data being uneven
C
	IF(XFAST)THEN
	    XST=XP(1)
	    IF(XDIR)THEN
		DO 200 I=1,NIY
		IF(XST.LT.XP(IPNT(I)))XST=XP(IPNT(I))
200		CONTINUE
		DO 205 I=1,NIX-1
		IF(X(I).GT.X(I+1))GOTO 206
205		CONTINUE
		I=NIX
206		NIX=I
	    ELSE
		DO 210 I=1,NIY
		IF(XST.GT.XP(IPNT(I)))XST=XP(IPNT(I))
210		CONTINUE
		DO 215 I=1,NIX-1
		IF(X(I).LT.X(I+1))GOTO 216
215		CONTINUE
		I=NIX
216		NIX=I
	    ENDIF
C
C	    Fix the IX and X arrays if uneven
C
	    DO 220 I=1,NIX
	    IF(XST.EQ.X(I))GOTO 221
220	    CONTINUE
221	    IF(I.EQ.1)GOTO 226
	    DO 225 J=I,NIX
	    JJ=J-I+1
	    X(JJ)=X(J)
225	    CONTINUE
226	    CONTINUE
	    NIX=NIX-I+1
C
C	    Setup the IXS array
C
	    DO 230 I=1,NIY
	    DO 235 J=IPNT(I),IPNT(I+1)
	    IF(XST.EQ.XP(J))GOTO 236
235	    CONTINUE
236	    IPS(I)=J-IPNT(I)
230	    CONTINUE
	ELSE
	    YST=YP(1)
	    IF(YDIR)THEN
		DO 300 I=1,NIX
		IF(YST.LT.YP(IPNT(I)))YST=YP(IPNT(I))
300		CONTINUE
		DO 305 I=1,NIY-1
		IF(Y(I).GT.Y(I+1))GOTO 306
305		CONTINUE
		I=NIY
306		NIY=I
	    ELSE
		DO 310 I=1,NIX
		IF(YST.GT.YP(IPNT(I)))YST=YP(IPNT(I))
310		CONTINUE
		DO 315 I=1,NIY-1
		IF(Y(I).LT.Y(I+1))GOTO 316
315		CONTINUE
		I=NIY
316		NIY=I
	    ENDIF
C
C	    Fix the IY and Y arrays if uneven
C
	    DO 320 I=1,NIY
	    IF(YST.EQ.Y(I))GOTO 321
320	    CONTINUE
321	    IF(I.EQ.1)GOTO 326
	    DO 325 J=I,NIY
	    JJ=J-I+1
	    Y(JJ)=Y(J)
325	    CONTINUE
326	    CONTINUE
	    NIY=NIY-I+1
C
C	    Setup the IYS array
C
	    DO 330 I=1,NIX
	    DO 335 J=IPNT(I),IPNT(I+1)
	    IF(YST.EQ.YP(J))GOTO 336
335	    CONTINUE
336	    IPS(I)=J-IPNT(I)
330	    CONTINUE
	ENDIF
C
C	Sort the slowly varying index
C
	IF(XFAST)THEN
405	    FLIP=.FALSE.
	    DO 400 J=2,NIY
	    IF(Y(J-1).GT.Y(J))THEN
		FLIP=.TRUE.
		YT=Y(J-1)
		Y(J-1)=Y(J)
		Y(J)=YT
		I=IPNT(J-1)
		IPNT(J-1)=IPNT(J)
		IPNT(J)=I
		I=IEND(J-1)
		IEND(J-1)=IEND(J)
		IEND(J)=I
		I=IPS(J-1)
		IPS(J-1)=IPS(J)
		IPS(J)=I
	    ENDIF
400	    CONTINUE
	    IF(FLIP)GOTO 405
	    YDIR=.TRUE.
	ELSE
415	    FLIP=.FALSE.
	    DO 410 J=2,NIX
	    IF(X(J-1).GT.X(J))THEN
		FLIP=.TRUE.
		XT=X(J-1)
		X(J-1)=X(J)
		X(J)=XT
		I=IPNT(J-1)
		IPNT(J-1)=IPNT(J)
		IPNT(J)=I
		I=IEND(J-1)
		IEND(J-1)=IEND(J)
		IEND(J)=I
		I=IPS(J-1)
		IPS(J-1)=IPS(J)
		IPS(J)=I
	    ENDIF
410	    CONTINUE
	    IF(FLIP)GOTO 415
	    XDIR=.TRUE.
	ENDIF
C
C	Begin interpolation  (double linear)
C
C
C	Set up limits for close enough 
C
	XERR=DX*1.E-4
	YERR=DY*1.E-4
	IF(XDIR)THEN
	    X1E=X(1)-XERR
	    XNE=X(NIX)+XERR
	ELSE
	    X1E=X(1)+XERR
	    XNE=X(NIX)-XERR
	ENDIF
C
	IF(YDIR)THEN
	    Y1E=Y(1)-YERR
	    YNE=Y(NIY)+YERR
	ELSE
	    Y1E=Y(1)+YERR
	    YNE=Y(NIY)-YERR
	ENDIF
C
	DO 50 J=1,NIX-1
	  ODX(J)=1./(X(J+1)-X(J))
50	CONTINUE
	DO 51 J=1,NIY-1
	  ODY(J)=1./(Y(J+1)-Y(J))
51	CONTINUE
C
	DO 101 J=1,NX   ! 100
	XJ=X1+DX*(J-1)
	DO 100 L=1,NY
	YL=Y1+DY*(L-1)
C
C	Find where Z(J,L) is located
C
	XIN=.FALSE.
	I=1
	IF(XDIR)THEN
	    IF(XJ.GE.X1E.AND.XJ.LE.XNE)THEN
		XIN=.TRUE.
		DO 110 I=2,NIX
 	IF(XJ-XERR.LE.X(I))GOTO 111
110		CONTINUE
111		I=I-1
	    ELSE
		IF(XJ.GT.X(NIX))I=NIX
	    ENDIF
	ELSE
	    IF(XJ.LE.X1E.AND.XJ.GE.XNE)THEN
		XIN=.TRUE.
		DO 120 I=2,NIX
		IF(XJ+XERR.GE.X(I))GOTO 121
120		CONTINUE
121		I=I-1
	    ELSE
		IF(XJ.LT.X(NIX))I=NIX
	    ENDIF
	ENDIF
C
	YIN=.FALSE.
	K=1
	IF(YDIR)THEN
	    IF(YL.GE.Y1E.AND.YL.LE.YNE)THEN
		YIN=.TRUE.
		DO 130 K=2,NIY
		IF(YL-YERR.LE.Y(K))GOTO 131
130		CONTINUE
131		K=K-1
	    ELSE
		IF(YL.GT.Y(NIY))K=NIY
	    ENDIF
	ELSE
	    IF(YL.LE.Y1E.AND.YL.GE.YNE)THEN
		YIN=.TRUE.
		DO 140 K=2,NIY
		IF(YL+YERR.GE.Y(K))GOTO 141
140		CONTINUE
141		K=K-1
	    ELSE
		IF(YL.LT.Y(NIY))K=NIY
	    ENDIF
	ENDIF
C
C	If inside then do the interpolation
C
	IF(XIN.AND.YIN)THEN
C
C	    Get the four corners
C
	    I00=IPOS(I  ,K  ,IPNT,IPS,IEND,XFAST,N)
	    I01=IPOS(I  ,K+1,IPNT,IPS,IEND,XFAST,N)
	    I10=IPOS(I+1,K  ,IPNT,IPS,IEND,XFAST,N)
	    I11=IPOS(I+1,K+1,IPNT,IPS,IEND,XFAST,N)
C
	    IF(ZP(I00).GE.BIG.OR.ZP(I10).GE.BIG.OR.
     *	    ZP(I01).GE.BIG.OR.ZP(I11).GE.BIG)THEN
		Z(J,L)=1.0E35
	    ELSE
		DXI=(XJ-X(I))*ODX(I)
		DYK=(YL-Y(K))*ODY(K)
C
C		Do it!
C
		Z(J,L)=ZP(I00)+(ZP(I10)-ZP(I00))*DXI
     *			      +(ZP(I01)-ZP(I00))*DYK
     *		+(ZP(I11)-ZP(I01)-ZP(I10)+ZP(I00))*DXI*DYK
	    ENDIF
	ELSE
	    Z(J,L)=1.0E35
C
	    IF(YIN.AND.ABS(DX).GT.ABS(XJ-X(I)))THEN
		I00=IPOS(I,K  ,IPNT,IPS,IEND,XFAST,N)
		I01=IPOS(I,K+1,IPNT,IPS,IEND,XFAST,N)
		IF(ZP(I00).LT.BIG.AND.ZP(I01).LT.BIG)THEN
		    A=(ZP(I01)-ZP(I00))*ODY(K)
		    B=ZP(I00)-A*Y(K)
		    Z(J,L)=A*YL+B
		ENDIF
	    ELSE IF(XIN.AND.ABS(DY).GT.ABS(YL-Y(K)))THEN
		I00=IPOS(I  ,K,IPNT,IPS,IEND,XFAST,N)
		I10=IPOS(I+1,K,IPNT,IPS,IEND,XFAST,N)
		IF(ZP(I00).LT.BIG.AND.ZP(I10).LT.BIG)THEN
		    A=(ZP(I10)-ZP(I00))*ODX(I)
		    B=ZP(I00)-A*X(I)
		    Z(J,L)=A*XJ+B
		ENDIF
	    ENDIF
	ENDIF

100	CONTINUE
101	CONTINUE

	RETURN
	END

	FUNCTION IPOS(I,J,IPNT,IPS,IEND,FAST,N)
	LOGICAL FAST
	DIMENSION IPNT(20000),IPS(20000),IEND(20000)
	IF(FAST)THEN
	    IPOS=IPNT(J)+I-1+IPS(J)
	    IF(IPOS.GE.IEND(J))IPOS=N+1
	ELSE
	    IPOS=IPNT(I)+J-1+IPS(I)
	    IF(IPOS.GE.IEND(I))IPOS=N+1
	ENDIF
	RETURN
	END
