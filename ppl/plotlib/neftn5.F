      SUBROUTINE NEFTN5(J,KM,IS) 
C** 
C**    @(#)neftn5.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
      COMMON/CEFTN5/ IN,IA,LT,NAME,INTEMP
      CHARACTER IN*80,IA*1,LT*1,NAME*81,INTEMP*80
C 
C 
C     WRITTEN IN MAY, 1980 BY DON DENBO IT IS A FTN5 
C     UTILITY ROUTINE FORM COMRD5. 
C 
C 
	IS=5
      J=J+1 
      IF(J.GT.KM)RETURN  
      IA=IN(J:J) 
      IF(IA.EQ.LT)RETURN  
	IS=3
      IF(IA.EQ.',')RETURN  
	IS=2
      IF(IA.EQ.' ')RETURN  
	IS=4
      IF(IA.EQ.'/')RETURN  
	IS=1
      IF(IA.GE.'+'.AND.IA.LE.'9')RETURN  
	IS=4
      RETURN  
      END 
