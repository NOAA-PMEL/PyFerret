      SUBROUTINE DELAY(ISEC) 
C** 
C**    @(#)delay.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C     SUBROUTINE TO DELAY EXECUTION BY APPROX ISEC OF REAL TIME 
#ifdef unix
	integer NULL
#else
	byte NULL
#endif
	CHARACTER ID*8
	DATA NULL/0/
      DIMENSION I(3) 
      CALL TIME(ID) 

#ifdef unix
      read(unit=id,fmt=100) i
#else
      DECODE(9,100,ID)I 
#endif

100   FORMAT(I2,1X,I2,1X,I2) 
      IST=I(1)*3600+I(2)*60+I(3) 
10    DO 20 K=1,100 
20    CALL CHOUT(NULL,1) 
      CALL CHDMP 
      CALL TIME(ID) 
#ifdef unix
      read(unit=id,fmt=100) i
#else
      DECODE(9,100,ID)I 
#endif

      ISP=I(1)*3600+I(2)*60+I(3) 
      IF(ISP-IST.LT.ISEC)GOTO 10 
      RETURN 
C**  THIS PROGRAM VALID ON FTN4 AND FTN5 ** 
      END 
