      FUNCTION NDIG(JDIG)
C** 
C**    @(#)ndig5.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C        NDIG FINDS THE NUMBER OF DIGITS IN AN INTEGER
      JXTRA=1
      IF(JDIG.GE.10)GO TO 200
      NDIG=1
      JXTRA=2
      IF(JDIG.GE.0)RETURN

  200 NDIG=IFIX(ALOG10(ABS(FLOAT(JDIG))))+JXTRA
      RETURN
C**  THIS PROGRAM VALID ON FTN4 AND FTN5 **
      END
