      SUBROUTINE SMY1(X,XFACT,YFACT,Y,YHI,YTIC,NSMTCY) 
C** 
C**    @(#)smy1.f	1.1    3/10/88
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
C     THIS SUBROUTINE PLOTS THE SMALL YTICS ON THE Y AXIS. 
C     THIS DIFERS FROM SMY IN THAT THE TICS ARE LARGER. 
C 
      IF(NSMTCY.LT.1) RETURN 
      SMTIC=YTIC/FLOAT(NSMTCY+1) 
      DO 10 I=1,NSMTCY 
          YTEMP=Y+FLOAT(I)*SMTIC 
          IF((YHI-YTEMP)/YTIC.GE.0) CALL PLOTT7(X,YTEMP,XFACT, 
     *YFACT,1,-5) 
   10     CONTINUE 
      RETURN 
C**  THIS PROGRAM VALID ON FTN4 AND FTN5 ** 
      END 
