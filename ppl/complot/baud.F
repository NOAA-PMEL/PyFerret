      SUBROUTINE BAUD(IRATE)
C** 
C**    @(#)baud.f	1.1    3/10/88
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
C     CALL BAUD(IRATE)
C
C     COMPLOT ASSUMES THAT A TEKTERMINAL IS OPERATING AT 2400
C     BAUD.  THIS FACT IS USED TO COMPUTE THE NUMBER OF CHARAC-
C     TERS THAT MUST BE TRANSMITTED TO THE DEVICE TO PRODUCE
C     THE APPROPRIATE ERASE DELAY TIME OR DRAWING TIME.  THIS
C     SUBROUTINE MAY BE CALLED TO RECALCULATE THE NUMBER OF
C     CHARACTERS IF A BAUD RATE OTHER THAN 2400 IS BEING USED.
C
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
C
C     CHECK FOR A LEGAL BAUD RATE.
C
      IF(IRATE .LT. 110 .OR. IRATE .GT. 9600)RETURN
      BAUDR = IRATE
      RETURN
C
      END
