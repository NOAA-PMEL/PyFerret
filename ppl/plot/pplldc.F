	SUBROUTINE PPLLDC(K,Z,MX,MY,IMN,IMX,JMN,JMX,
     *		PI,PJ,NX1,NY1,XMIN1,YMIN1,DX1,DY1)
	REAL PI(*),PJ(*),Z(MX,MY)
C** 
C**    @(#)pplldc.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C              Mod *sh* 4.1.91 Bug fix in PPLLDV fixed, see below
C              Mod *sh* 5.15.91 dx,dy put into common block
C              Mod *sh* 3.7.97 *sh* replace limits if curvilinear plot
c
c	k	=0 already on an equally spaced rectangular grid
c		=1 on a unequally spaced, but rectangular grid
c	z	input array
c	mx	dimensioned x size of z(mx,my)
c	my	dimensioned y size of z
c	imn	initial x index value
c	imx	final x index value
c	jmn	initial y index value
c	jmx	final y index value
c	pi	x positions pi(mx)
c	pj	y positions pj(my)
c	nx1	number of x grid points in output buffer
c	ny1	number of y grid points in output buffer
c	xmin1	position of 1,1 in output grid
c	ymin1	position of 1,1 in output grid
c	dx	spacing of x grid points in output buffer
c	dy	spacing of y grid points in output buffer
C
C
C
#ifdef unix
	INCLUDE 'pplv11inc/PARAMPL5.DAT'
	INCLUDE 'pplv11inc/ZGRID.INC'
	INCLUDE 'pplv11inc/ZGRIDD.INC'
	INCLUDE 'pplv11inc/MISS.INC'
	INCLUDE 'pplv11inc/LINES.INC'
	INCLUDE 'pplv11inc/PPLDAT.INC'
	INCLUDE 'pplv11inc/HD.INC'
	INCLUDE 'pplv11inc/DATA.INC'
#else
        INCLUDE 'pplv11inc:PARAMPL5.DAT'
        INCLUDE 'pplv11inc:ZGRID.INC'
        INCLUDE 'pplv11inc:ZGRIDD.INC'
        INCLUDE 'pplv11inc:MISS.INC'
        INCLUDE 'pplv11inc:LINES.INC'
        INCLUDE 'pplv11inc:PPLDAT.INC'
        INCLUDE 'pplv11inc:HD.INC'
        INCLUDE 'pplv11inc:DATA.INC'
#endif
	LOGICAL		ITS_CURVILINEAR
	REAL		xmn, xmx, ymn, ymx


	COMMON/LDDXDY/ DX,DY
C
	call rsmnmx
	JTYPE=0
	NX=NX1
	NY=NY1
	DX=DX1
	DY=DY1
	XMIN=XMIN1
	YMIN=YMIN1
	IF(K.EQ.1)THEN
	    JTYPE=2
	    XMIN=PI(1)
	    XMAX=PI(MX)
	    YMIN=PJ(1)
	    YMAX=PJ(MY)
	    NX=MX
	    NY=MY
	    DO 100 I=IMN,IMX
	    DO 100 J=JMN,JMX
100		X(NX*(J-1)+I)=Z(I,J)
	    IV=NX*NY
	    DO 101 I=1,NX
101		X(IV+I)=PI(I)
	    IV=IV+NX
	    DO 102 J=1,NY
102		X(IV+J)=PJ(J)
C
	ELSE
C
C	DATA ALREADY GRIDDED
C
	    NX=IMX-IMN+1
	    NY=JMX-JMN+1
	    XMAX=XMIN+DX*(NX-1)
	    YMAX=YMIN+DY*(NY-1)
	    I=0
	    DO 400 II=IMN,IMX
	    I=I+1
	    J=0
	    DO 400 JJ=JMN,JMX
	    J=J+1
	    X(NX*(J-1)+I)=Z(II,JJ)
400	    CONTINUE
	ENDIF
	linen=0

* for a curvilinear plot the apparent min/max, which are in fact the min/max
* index values, are wrong -- set symbols based on the true axis ranges
	IF ( ITS_CURVILINEAR() ) THEN
	   CALL GET_CURVILINEAR_LIMITS( xmn, xmx, ymn, ymx )
	   call stmnmx(x,nsize,xmn,xmx,ymn,ymx)
	ELSE
	   call stmnmx(x,nsize,xmin,xmax,ymin,ymax)
	ENDIF
	RETURN
        end

	SUBROUTINE PPLLDV(K,Z,MX,MY,IMN,IMX,JMN,JMX)
!	ENTRY PPLLDV(K,Z,MX,MY,IMN,IMX,JMN,JMX)
! ... start of copied from pplldc
#ifdef unix
	INCLUDE 'pplv11inc/PARAMPL5.DAT'
	INCLUDE 'pplv11inc/ZGRID.INC'
	INCLUDE 'pplv11inc/ZGRIDD.INC'
	INCLUDE 'pplv11inc/MISS.INC'
	INCLUDE 'pplv11inc/LINES.INC'
	INCLUDE 'pplv11inc/PPLDAT.INC'
	INCLUDE 'pplv11inc/HD.INC'
	INCLUDE 'pplv11inc/DATA.INC'
#else
        INCLUDE 'pplv11inc:PARAMPL5.DAT'
        INCLUDE 'pplv11inc:ZGRID.INC'
        INCLUDE 'pplv11inc:ZGRIDD.INC'
        INCLUDE 'pplv11inc:MISS.INC'
        INCLUDE 'pplv11inc:LINES.INC'
        INCLUDE 'pplv11inc:PPLDAT.INC'
        INCLUDE 'pplv11inc:HD.INC'
        INCLUDE 'pplv11inc:DATA.INC'
#endif
	REAL Z(MX,MY)
	COMMON/LDDXDY/ DX,DY
! ... end of copied from pplldc

C
C	LOAD THE SECOND SET OF GRID INFORMATION FOR VECTOR PLOTS
C
	IF(.NOT.(JTYPE.EQ.0.OR.JTYPE.EQ.2))RETURN
	call rsmnmx
	JTYPE=-1
	IVLEN=NX*NY
	IF(K.EQ.1)THEN
	    JTYPE=-2
	    IVLEN=NX*NY+NX+NY
	    DO 500 I=IMN,IMX
	    DO 500 J=JMN,JMX
500	    X(IVLEN+NX*(J-1)+I)=Z(I,J)
	    IV=IVLEN+NX*NY
	    DO 501 I=1,NX
!501	    X(IV+I)=PI(I)      ! illegal use of PI (not passed)
501	    X(IV+I)=x(NX*NY+i) ! reuse coords from PPLLDC
	    IV=IV+NX
	    DO 502 J=1,NY
!502	    X(IV+J)=PJ(J)      ! illegal use of PJ (not passed)
502	    X(IV+J)=x(NX*NY+nx+j) ! reuse coords from PPLLDC
	ELSE
C
C	DATA ALREADY GRIDDED
C
	    NX=IMX-IMN+1
	    NY=JMX-JMN+1
	    XMAX=XMIN+DX*(NX-1)
	    YMAX=YMIN+DY*(NY-1)
	    I=0
	    DO 700 II=IMN,IMX
	    I=I+1
	    J=0
	    DO 700 JJ=JMN,JMX
	    J=J+1
	    X(IVLEN+NX*(J-1)+I)=Z(II,JJ)
700	    CONTINUE
	ENDIF
	linen=0
	call stmnmx(x,nsize,xmin,xmax,ymin,ymax)
	RETURN
	END
