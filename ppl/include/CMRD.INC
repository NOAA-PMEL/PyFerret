c
c	cmrd
C** 
C**    @(#)CMRD.INC	1.1    3/10/88
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
	COMMON/CMRD/ P(20),M,N,IFLG(20),ISLEN,LBSTRT
c
c	p(1:20)	numerical parameters parsed by COMRD5
c	m	number of numerical parameters 
c	n	number of characters in LABEL
c	iflg(1:20)	=0 corresponding parameter not set by COMRD5
c			=1 corresponding parameter is set by COMRD5
c	islen	length of STRING  (getcmd)
c	lbstrt	start position of LABEL in STRING
c
c*******************************************************************
