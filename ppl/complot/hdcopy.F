      SUBROUTINE HDCOPY
C** 
C**    @(#)hdcopy.f	1.1    3/10/88
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
C     CALL HDCOPY
C
C     THIS SUBROUTINE ACTIVATES THE HARD-COPY ACCESSORY FOR THE
C     TEKTERMINAL.
C
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
C
C     ACTIVATE THE HARD-COPY ACCESSORY.
C
      CALL CHOUT(ESC,1)
      CALL CHOUT(23,1)
C
C     TRANSMIT SYNC CHARACTERS WHILE THE HARD COPY IS PRODUCED.
C
      NDELAY = IFIX(10.0 * FLOAT(BAUDR / 10))
      DO 10 J=1,NDELAY
   10 CALL CHOUT(SYN,1)
C
C     WRITE OUT THE CHARACTER BUFFER.
C
      CALL CHDMP
      RETURN
C
      END
