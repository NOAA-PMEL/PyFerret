      INTEGER*4 FUNCTION ITDIFF(ITM1,ITM2) 
C** 
C**    @(#)itdiff.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C     THIS FUNCTION RETURNS THE DIFFERENCE IN DAYS AND MINUTES 
C     BETWEEN ITM1 AND ITM2.  LAST 5 DIGITS IS THE SECOND COUNT 
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
	CHARACTER*(*) ITM1,ITM2
      CALL TIME3(ITM1,IY1,IMN1,ID1,IH1,IM1) 
      CALL TIME3(ITM2,IY2,IMN2,ID2,IH2,IM2) 
      ITDIFF=(IDAY(IY2,IMN2,ID2)-IDAY(IY1,IMN1,ID1))*10000+ 
     *(IH2-IH1)*60+(IM2-IM1) 
      RETURN 
      END 
