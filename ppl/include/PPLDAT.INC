c
c
!	PARAMETER (NSIZE=150000)
!	Changed back to 200K J Davison
!	Now 500 000 J Davison 3.4.93
!	Now double it to 1000000
	PARAMETER (NSIZE=1000000)
c
c	ppldat
C** 
C**    @(#)PPLDAT.INC	1.1    3/10/88
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
	COMMON/PPLDTA/X(NSIZE),IBASE,PBUF
	INTEGER IBASE,PBUF
c
c	x()	plot buffer (data storage)
c	ibase	current position in buffer for next data set
c	pbuf	=0 buffer empty
c		=1 buffer not empty
c
c*******************************************************************
