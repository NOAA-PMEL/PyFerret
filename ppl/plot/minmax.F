	SUBROUTINE STMNMX(X,NSIZE,XMIN1,XMAX1,YMIN1,YMAX1)
	DIMENSION X(NSIZE)
C** 
C**    @(#)minmax.F	1.1    3/10/88
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
	INCLUDE 'pplinc/LINES.INC'
	INCLUDE 'pplinc/DATA.INC'
	INCLUDE 'pplinc/MISS.INC'
	include 'pplinc/HD.INC'
#else
        INCLUDE 'pplinc:PARAMPL5.DAT'
        INCLUDE 'pplinc:LINES.INC'
        INCLUDE 'pplinc:DATA.INC'
        INCLUDE 'pplinc:MISS.INC'
	include 'pplinc:HD.INC'
#endif
	CHARACTER SYM*30,VAL*10
C
	SYM='*PPL$LINE_COUNT'
	WRITE(VAL,'(I2)')LINEN
	CALL PUTSYM(SYM,VAL,2,IER)
C
	IF(LINEN.EQ.0)THEN
C
C	CONTOUR FORM
C
	    SYM='*PPL$XMIN'
	    CALL PUTVAL(SYM,XMIN1,4,IER)
C
	    SYM='*PPL$XMAX'
	    CALL PUTVAL(SYM,XMAX1,4,IER)
C
	    SYM='*PPL$YMIN'
	    CALL PUTVAL(SYM,YMIN1,4,IER)
C
	    SYM='*PPL$YMAX'
	    CALL PUTVAL(SYM,YMAX1,4,IER)
C
c	find the min and max and the locations of them
c
	    ZMIN=1.E38
            ZMAX=-ZMIN
	    jmn=1
	    jmx=1
	    imx=1
	    imn=1
	    if(jtype.eq.-1)then
		ivlen=nx*ny
	    else
		ivlen=nx*ny+nx+ny
	    endif
            DO 300 J=1,NY
            DO 300 I=1,NX
		if(jtype.ge.0)then
		    XT=X(NX*(J-1)+I)
		else
		    xtt1=x(nx*(j-1)+i)
		    xtt2=x(ivlen+nx*(j-1)+i)
		    if (xtt1.ge.1.e35 .or. xtt2.ge.1.e35) goto 300
		    xt=sqrt(xtt1**2+xtt2**2)
		endif
        	IF(ZLE.AND.XT.LE.CMZLE)GOTO 300
        	IF(ZEQ.AND.XT.EQ.CMZEQ)GOTO 300
        	IF(ZGE.AND.XT.GE.CMZGE)GOTO 300
        	IF(XT.LT.ZMIN)then
		    ZMIN=XT
		    jmn=j
		    imn=i
		endif
        	IF(XT.GT.ZMAX)then
		    ZMAX=XT
		    jmx=j
		    imx=i
		endif
300         CONTINUE
	    if(abs(jtype).eq.2)then
		ivlen=nx*ny
		xhigh=x(ivlen+imx)
		xlow=x(ivlen+imn)
c
		ivlen=ivlen+nx
		yhigh=x(ivlen+jmx)
		ylow=x(ivlen+jmn)
	    else
		dx=(xmax-xmin)/(nx-1)
		xhigh=xmin+dx*(imx-1)
		xlow=xmin+dx*(imn-1)
c
		dy=(ymax-ymin)/(ny-1)
		yhigh=ymin+dy*(jmx-1)
		ylow=ymin+dy*(jmn-1)
	    endif
c
	    SYM='*PPL$ZMIN'
	    CALL PUTVAL(SYM,ZMIN,4,IER)
C
	    SYM='*PPL$ZMAX'
	    CALL PUTVAL(SYM,ZMAX,4,IER)
c
	    sym='*PPL$XHIGH'
	    call putval(sym,xhigh,4,ier)
c
	    sym='*PPL$YHIGH'
	    call putval(sym,yhigh,4,ier)
c
	    sym='*PPL$XLOW'
	    call putval(sym,xlow,4,ier)
c
	    sym='*PPL$YLOW'
	    call putval(sym,ylow,4,ier)
	ELSE
C
C	LINE DATA
C
	    WRITE(SYM,998)'XMIN',LINEN
998	    FORMAT('*PPL$',A,'(',I2,')')
	    CALL PUTVAL(SYM,XMIN1,4,IER)
C
	    WRITE(SYM,998)'XMAX',LINEN
	    CALL PUTVAL(SYM,XMAX1,4,IER)
C
	    WRITE(SYM,998)'YMIN',LINEN
	    CALL PUTVAL(SYM,YMIN1,4,IER)
C
	    WRITE(SYM,998)'YMAX',LINEN
	    CALL PUTVAL(SYM,YMAX1,4,IER)
C
C	SET FIRST - LAST VALUES
C
	    IBASEX=0
	    IBASEY=NSIZE/2
	    DO 10 I=1,LINEN-1
	    LEN=LLENG(I)
	    IBASEX=IBASEX+LEN
	    IBASEY=IBASEY+LEN
10	    CONTINUE
	    LEN=LLENG(LINEN)
	    DO 20 J=1,LEN
	    IF((XLE.AND.X(IBASEX+J).LE.CMXLE).OR.
     *		(XEQ.AND.X(IBASEX+J).EQ.CMXEQ).OR.
     *		(XGE.AND.X(IBASEX+J).GE.CMXGE).OR.
     *		(YLE.AND.X(IBASEY+J).LE.CMYLE).OR.
     * 	(YEQ.AND.X(IBASEY+J).EQ.CMYEQ).OR.
     *		(YGE.AND.X(IBASEY+J).GE.CMYGE))GOTO 20
	    XT=X(IBASEX+J)*XFCT(I)+XOFF(I)
	    YT=X(IBASEY+J)*YFCT(I)+YOFF(I)
	    WRITE(SYM,998)'XFIRST',LINEN
	    CALL PUTVAL(SYM,XT,4,IER)
	    WRITE(SYM,998)'YFIRST',LINEN
	    CALL PUTVAL(SYM,YT,4,IER)
	    GOTO 30	    
20	    CONTINUE
30	    DO 40 J=LEN,1,-1
	    IF((XLE.AND.X(IBASEX+J).LE.CMXLE).OR.
     *		(XEQ.AND.X(IBASEX+J).EQ.CMXEQ).OR.
     *		(XGE.AND.X(IBASEX+J).GE.CMXGE).OR.
     *		(YLE.AND.X(IBASEY+J).LE.CMYLE).OR.
     *		(YEQ.AND.X(IBASEY+J).EQ.CMYEQ).OR.
     *		(YGE.AND.X(IBASEY+J).GE.CMYGE))GOTO 40
	    XT=X(IBASEX+J)*XFCT(I)+XOFF(I)
	    YT=X(IBASEY+J)*YFCT(I)+YOFF(I)
	    WRITE(SYM,998)'XLAST',LINEN
	    CALL PUTVAL(SYM,XT,4,IER)
	    WRITE(SYM,998)'YLAST',LINEN
	    CALL PUTVAL(SYM,YT,4,IER)
	    GOTO 50
40	    CONTINUE
	ENDIF
50	RETURN
	END
	SUBROUTINE RSMNMX
C
C	RESET MIN MAX SYMBOLS AND SET LINE_COUNT TO ZERO
C
#ifdef unix
	INCLUDE 'pplinc/PARAMPL5.DAT'
	INCLUDE 'pplinc/LINES.INC'
#else
        INCLUDE 'pplinc:PARAMPL5.DAT'
        INCLUDE 'pplinc:LINES.INC'
#endif
	CHARACTER SYM*30,TEMP*2
C
	SYM='*PPL$LINE_COUNT'
	TEMP='0'
	CALL PUTSYM(SYM,TEMP,1,IER)
C
	DO 10 I=1,LNUM
	WRITE(SYM,999)'XMIN',I
999	FORMAT('*PPL$',A,'(',I2,')')
	CALL DELSYM(SYM,IER)
	WRITE(SYM,999)'XMAX',I
	CALL DELSYM(SYM,IER)
	WRITE(SYM,999)'YMIN',I
	CALL DELSYM(SYM,IER)
	WRITE(SYM,999)'YMAX',I
	CALL DELSYM(SYM,IER)
C
	WRITE(SYM,999)'XFIRST',I
	CALL DELSYM(SYM,IER)
	WRITE(SYM,999)'XLAST',I
	CALL DELSYM(SYM,IER)
	WRITE(SYM,999)'YFIRST',I
	CALL DELSYM(SYM,IER)
	WRITE(SYM,999)'YLAST',I
	CALL DELSYM(SYM,IER)
10	CONTINUE
C
	SYM='*PPL$XMIN'
	CALL DELSYM(SYM,IER)
	SYM='*PPL$XMAX'
	CALL DELSYM(SYM,IER)
	SYM='*PPL$YMIN'
	CALL DELSYM(SYM,IER)
	SYM='*PPL$YMAX'
	CALL DELSYM(SYM,IER)
	SYM='*PPL$ZMIN'
	CALL DELSYM(SYM,IER)
	SYM='*PPL$ZMAX'
	CALL DELSYM(SYM,IER)
	sym='*PPL$XHIGH'
	CALL DELSYM(SYM,IER)
	sym='*PPL$YHIGH'
	CALL DELSYM(SYM,IER)
	sym='*PPL$XLOW'
	CALL DELSYM(SYM,IER)
	sym='*PPL$YLOW'
	CALL DELSYM(SYM,IER)
	RETURN
	END
