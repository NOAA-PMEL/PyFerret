      SUBROUTINE SMX1(X,XHI,XTIC,NSMTCX,XFACT,YFACT,Y) 
C** 
C**    @(#)smx1.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C 
C     THIS SUBROUTINE PLOTS THE SMALL XTICS ON THE X AXIS. 
C     IT DIFFERS FROM SMX IN THAT THE TICS ARE BIGGER. 
C 
      IF(NSMTCX.LT.1) RETURN 
      SMTIC=XTIC/FLOAT(NSMTCX+1) 
      DO 10 I=1,NSMTCX 
          XTEMP=X+FLOAT(I)*SMTIC 
          IF((XHI-XTEMP)/XTIC.GE.0) CALL PLOTT7(XTEMP,Y,XFACT, 
     *YFACT,1,-7) 
   10     CONTINUE 
      RETURN 
C**  THIS PROGRAM VALID ON FTN4 AND FTN5 ** 
      END 
