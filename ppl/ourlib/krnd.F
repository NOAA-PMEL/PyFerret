      FUNCTION KRND(A)
C** 
C**    @(#)krnd.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
      KRND=INT(SIGN(ABS(A)+0.5,A))
      RETURN
C**  THIS PROGRAM VALID ON FTN4 AND FTN5 **
      END
