      FUNCTION IDAY(IY,IM,ID) 
C** 
C**    @(#)iday.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
      IF(IM.GT.2)THEN 
      YR=FLOAT(IY) 
      XM=FLOAT(IM+1) 
      ELSE 
      YR=FLOAT(IY-1) 
      XM=FLOAT(IM+13) 
      END IF 
      IDAY=INT(365.25*YR)+INT(30.6001*XM)+ID 
      RETURN 
      END 
