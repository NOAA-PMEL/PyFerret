	SUBROUTINE PLOTZ(Z,XF,YF,XLPLOT,YLPLOT,XHPLOT,YHPLOT)
	DIMENSION Z(NX,NY)
C**
C**    @(#)plotz.F	1.4    9/9/88
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
	INCLUDE 'pplinc/PARAMPL5.DAT'
	INCLUDE 'pplinc/ZGRIDD.INC'
	INCLUDE 'pplinc/ZGRID.INC'
	INCLUDE 'pplinc/AXIS.INC'
	INCLUDE 'pplinc/CONT.INC'
	INCLUDE 'pplinc/HD.INC'
	INCLUDE 'pplinc/TAXIS.INC'
	INCLUDE 'pplinc/AXISL.INC'
	INCLUDE 'pplinc/PEN.INC'
	INCLUDE 'pplinc/LINES.INC'
#else
	INCLUDE 'pplinc:PARAMPL5.DAT'
	INCLUDE 'pplinc:ZGRIDD.INC'
	INCLUDE 'pplinc:ZGRID.INC'
	INCLUDE 'pplinc:AXIS.INC'
	INCLUDE 'pplinc:CONT.INC'
	INCLUDE 'pplinc:HD.INC'
	INCLUDE 'pplinc:TAXIS.INC'
	INCLUDE 'pplinc:AXISL.INC'
	INCLUDE 'pplinc:PEN.INC'
	INCLUDE 'pplinc:LINES.INC'
#endif
C
C	VAX/VMS
C
	DIMENSION IZMRKD(3)
	DATA IZMRKD/12,16,3/
C
C	END VAX/VMS
C
C     DETERMINE CONTOUR LEVELS
	IF(IAUTOC.EQ.1)THEN
	    IF(ZMIN.EQ.ZMAX)THEN
		NLEV2=0
	    ELSE
		CALL RANGE(ZMIN,ZMAX,NLEV,ZNP,ZMP,DZ)
		NLEV2=(ZMP-ZNP)/DZ+1
		JJ=NSIGC-IFIX(ALOG10(ABS(ZMP-ZNP))+.5)
		IF(JJ.LE.0)JJ=-1
		DO 70 I=1,NLEV2
		    LWGT(I)=2
		    LDIG(I)=-1
		    ZLEV(I)=ZNP+DZ*FLOAT(I-1)
		    IF(ABS(ZLEV(I)).LT.1.E-5*DZ)GOTO 70
		    IF(ZLEV(I).GT.0.)THEN
			LWGT(I)=1
		    ELSE
			LWGT(I)=3
		    END IF
		    LDIG(I)=JJ
70		CONTINUE
	    ENDIF
	ELSE
	    IF(ZMIN.EQ.ZMAX)THEN
		NLEV2=0
	    ELSE
		NLEV2=NLEV
	    ENDIF
	END IF
C     DRAW CONTOUR
C
C	VAX/VMS
C
        if(abs(jtype).eq.2 .and. iztyp.ne.0)then
           call markh(hmark(1))
           if(izmrk.eq.0)then
              imrk=izmrkd(iztyp)
           else
              imrk=izmrk
           endif
           if(iztyp.eq.1)then
              do 600 i=1,nx
                 call trans(0,z(i,ny+1),ylo,xt,y)
                 call plot(xt,y,0,imrk)
                 call mark(imrk)
 600          continue
           else if(iztyp.eq.2)then
              do 610 j=1,ny
                 call trans(0,xlo,z(j,ny+2),x,yt)
                 call plot(x,yt,0,imrk)
                 call mark(imrk)
 610          continue
           else
              do 620 i=1,nx
                 do 630 j=1,ny
                    call trans(0,z(i,ny+1),z(j,ny+2),xt,yt)
                    call plot(xt,yt,0,imrk)
                    call mark(imrk)
 630             continue
 620          continue
           endif
        endif
	IF(IZLEN.NE.0.AND.IZTYP.NE.0)THEN
	    CALL MARKH(HMARK(1))
	    IF(IZMRK.EQ.0)THEN
		IMRK=IZMRKD(IZTYP)
	    ELSE
		IMRK=IZMRK
	    ENDIF
	    IF(IZTYP.EQ.1)THEN
		call trans(0,xx(1),ylo,xt,y)
		CALL PLOT(XT,Y,0,IMRK)
		CALL MARK(IMRK)
		DO 100 I=2,IZLEN
		    IF(XT.NE.XX(I))THEN
			call trans(0,xx(i),ylo,xt,y)
			CALL PLOT(xt,Y,0,IMRK)
			CALL MARK(IMRK)
		    ENDIF
100		CONTINUE
	    ELSE IF(IZTYP.EQ.2)THEN
		call trans(0,xlo,yy(1),x,yt)
		CALL PLOT(X,YT,0,IMRK)
		CALL MARK(IMRK)
		DO 200 I=2,IZLEN
		    IF(YT.NE.YY(I))THEN
			call trans(0,xlo,yy(i),x,yt)
			CALL PLOT(X,yt,0,IMRK)
			CALL MARK(IMRK)
		    ENDIF
200		CONTINUE
	    ELSE
		DO 300 I=1,IZLEN
		    call trans(0,xx(i),yy(i),xt,yt)
		    CALL PLOT(xt,yt,0,IMRK)
		    CALL MARK(IMRK)
300		CONTINUE
	    ENDIF
	ENDIF
C
C	END VAX/VMS
C
	CALL SCALE(1.,1.,XORG,YORG,0.,0.)
	IF(IWIND.NE.0)CALL WINDOW(0.0,0.0,XLEN,YLEN)
	CALL COLOR(IPEN(1))
C
C
	call trans(0,xmin,ymin,xtm,ytm)
	call trans(0,xmax,ymax,xtx,ytx)
	XFF = (XHPLOT - XLPLOT)/(xtx-xtm)
	XOF = XLPLOT - XFF*xtm
	YFF = (YHPLOT - YLPLOT)/(ytx-ytm)
	YOF = YLPLOT - YFF*ytm
C
	IF(abs(jtype).eq.2)then
C
	    do 500 i=1,nx
		call trans(0,z(i,ny+1),ymin,xx(izlen+i),yt)
500	    continue
	    do 501 j=1,ny
		call trans(0,xmin,z(j,ny+2),xt,yy(izlen+j))
501	    continue
	ELSE
C
	    DX = (xtx - xtm)/(NX - 1)
	    DY = (ytx - ytm)/(NY - 1)
C
C	STORE POSITIONS IN XX AND YY
C
	    DO 400 I=1,NX
400	    XX(IZLEN+I)=DX*(I-1)+xtm
	    DO 401 J=1,NY
401	    YY(IZLEN+J)=DY*(J-1)+ytm
	ENDIF
	CALL CONSEG(Z,NX,NY,XX(IZLEN+1),YY(IZLEN+1),NX,NY,
     *	XFF,XOF,YFF,YOF,ZLEV,LDIG,LWGT,LCLR,NLEV2,HGT,NARC,CANGLE)
C
C	END VAX/VMS
C
	CALL COLOR(IPEN(0))
	CALL SCALE(XF,YF,XORG,YORG,XLO,YLO)
	RETURN
	END
