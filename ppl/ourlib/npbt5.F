      FUNCTION NPBT(MTIME2,MTIME1,NMBP,JFLAG)
C** 
C**    @(#)npbt5.f	1.1    3/10/88
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
C         NPBT RETURNS THE NUMBER OF DATA POINTS BETWEEN TWO
C         MILITARY TIMES, GIVEN THE SAMPLING RATE IN MINUTES.
C
C         MTIME2: THE LATER OF THE TWO TIMES.
C         MTIME1: THE EARLIER OF THE TWO TIMES.
C         NMBP:   THE NUMBER OF MINUTES BETWEEN POINTS(SAMPLING RATE).
C         JFLAG:  0: COMPUTE THE NUMBER OF WHOLE SAMPLE INTERVALS
C                 BETWEEN THE TIMES.
C                 1: COMPUTE THE NUMBER OF WHOLE SAMPLE INTERVALS
C                 THAT INCLUDES THE TIMES.

C
	REAL MTIME1(3),MTIME2(3)
	INTEGER*4 MTMIN,MNDF
      MNDF=MTMIN(MTIME2)-MTMIN(MTIME1)
      NPBT=MNDF/NMBP
      IF(JFLAG.EQ.1) NPBT=NPBT+1
      RETURN
      END
