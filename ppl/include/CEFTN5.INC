c
c	ceftn5
C** 
C**    @(#)CEFTN5.INC	1.1    3/10/88
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
	COMMON/CEFTN5/ IN,IA,LT,CNAME,INTEMP
	CHARACTER IN*80,IA*1,LT*1,CNAME*81,INTEMP*80
c
c	in	last physical line read using COMRD5
c	ia	last character processed using COMRD5
c	lt	logical end of input character
c	cname	file name used for input
c	intemp	last logical line read using COMRD5
c
c*******************************************************************
