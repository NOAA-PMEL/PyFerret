c
c	dsfsve
C** 
C**    @(#)DSFSVE.INC	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c
	CHARACTER NAME(4)*81
	common/dsfnme/ name
	COMMON/DSFSVE/ INXT(4),IBKWD(4),IKEY(4)
c
c	dname(1:81,1:4)	names of DSF formatted files
c	inxt(1:4)	forward pointers for next header to read
c	ibkwd(1:4)	backward pointers for last header read
c
c*******************************************************************
