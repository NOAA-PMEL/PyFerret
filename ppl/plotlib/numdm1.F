      FUNCTION NUMDM1(IT) 
C** 
C**    @(#)numdm1.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C     THIS FUNCTION RETURNS NO. OF DAYS IN CURRENT MONTH
 	CHARACTER*10 IT
      DIMENSION ID(12) 
      DATA ID/31,28,31,30,31,30,31,31,30,31,30,31/ 
      CALL TIME3(IT,IY,IM,IDD,IH,IX) 
      NUMDM1=ID(IM) 
      IF(IM.EQ.2.AND.IY.EQ.4*(IY/4))NUMDM1=29
      RETURN 
      END 
