c
c	miss
C** 
C**    @(#)MISS.INC	1.1    3/10/88
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
	COMMON/MISS/CMXLE,CMXEQ,CMXGE,CMYLE,CMYEQ,CMYGE,
     *		    CMZLE,CMZEQ,CMZGE,
     *		      XLE,  XEQ,  XGE,  YLE,  YEQ,  YGE,
     *		      ZLE,  ZEQ,  ZGE
	LOGICAL XLE,XEQ,XGE,YLE,YEQ,YGE,ZLE,ZEQ,ZGE
c
c	cmxle	value for good data check for xle test(user units)
c	cmxeq	value for good data check for xeq test(user units)
c		...etc...
c	xle	test for x .le. cmxle for good data
c	xeq	test for x .eq. cmxeq for good data
c		...etc...
c
c*******************************************************************
