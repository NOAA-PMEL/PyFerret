	FUNCTION LENC(LAB,NC)
C** 
C**    @(#)lenc.f	1.1    3/10/88
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
	CHARACTER LAB*(*)
	DO 20 I=NC,1,-1
	IF(LAB(I:I).NE.' ')GOTO 21
20	CONTINUE
	LENC=0
	RETURN
21	LENC=I
	RETURN
	END
