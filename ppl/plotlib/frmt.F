      SUBROUTINE FRMT(XMIN,XMAX,IPRES,IF) 
C** 
C**    @(#)frmt.f	1.1    3/10/88
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
      CHARACTER*(*) IF 
C     THIS SUBROUTINE CHOOSE THE FORMAT   NOT ANSI STANDARD 
      DX=ABS(XMAX-XMIN) 
      XX=ABS(XMAX) 
      IF(XX.LT.ABS(XMIN))XX=ABS(XMIN) 
      IF(XX.GE.1000..OR.XX.LT..1)GOTO 100 
200   IP=ABS(IPRES) 
      ID=IP-IFIX(ALOG10(DX)+.5) 
      IF(ID.LT.0)ID=0 
      IB=IFIX(ALOG10(XX*1.00001)) 
      WRITE(IF,999)IB+ID+3,ID 
999   FORMAT('(F',I2,'.',I1,')') 
      RETURN 
100   IF(IPRES.LT.0)GOTO 200 
      ID=IPRES-1 
      IF(ID.LT.0)ID=0 
      WRITE(IF,998) ID+7,ID 
998   FORMAT('(1PE',I2,'.',I1,')') 
      RETURN 
      END 
