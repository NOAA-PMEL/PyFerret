	SUBROUTINE DEFLT(FILE,NME)
C** 
C**    @(#)deflt.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
	CHARACTER FILE*(*),NME*(*),TEMP*81
	IF(INDEX(FILE,':').NE.0)RETURN
	K=INDEX(NME,':')
	IF(K.EQ.0)RETURN
	TEMP=NME(1:K)
	TEMP(K+1:)=FILE
	FILE=TEMP
	RETURN
	END
