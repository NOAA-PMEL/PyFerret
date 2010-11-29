c
c	hd
C** 
C**    @(#)HD.INC	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
* V630  *acm* 9/09 Introduction of syntax for variance-based and histogram levels

	COMMON/HD/ XPTS,ZMIN,ZMAX,ZMEAN,ZSTD,XMIN,XMAX,NX,YMIN,YMAX,
     *	NY,JTYPE,JLAB(16),NCH,IXLB(16),NCHX,IYLB(16),NCHY
	DIMENSION IPTS(NMIPTS)
	INTEGER*2 JLAB,IXLB,IYLB
	INTEGER*4 IPTS
	EQUIVALENCE (IPTS(1),XPTS)
c
c	DSF header for FORMAT= DSF files.  (note: not BIBO files)
c
c	xpts	actual number of points in data file (y and x) (real)
c	zmin	minimum value for x and y  (real)
c	zmax	maximum value for x and y  (real)
c	zmean	mean for x and y  (real)
c       zstd	std for xy array
c	xmin	minimum x value  (real) (user units)
c	xmax	maximum x value  (real) (user units)
c	nx	number of x grid points (integer)
c	ymin	minimum y value (real) (user units)
c	ymax	maximum y value (real) (user units)
c	ny	number of y grid points (integer)
c	jtype	=0 2-d set (contour), =1 1-d set   (integer)
c	jlab(16)main label hollerith (integer)
c	nch	number of characters in jlab  (integer)
c	ixlb(16)x axis label hollerith (integer)
c	nchx	number of characters in ixlb  (integer)
c	iylb(16)y axis label hollerith  (integer)
c	nchy	number of characters in iylb  (integer)
c
c*******************************************************************
C
C	RSX-11M OR INT*2 DEFAULT
C
C      COMMON/HD/ IPTS(70) 
C	INTEGER*2 IPTS
C
C	END RSX-11M
C
C
C	VAX/VMS  OR INT*4 DEFAULT
C
C	COMMON/HD/ IPTS(38)
C	INTEGER*4 IPTS
C
C	END VAX/VMS
C
