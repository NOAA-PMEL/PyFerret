      SUBROUTINE MARKH(HEIGHT)
C** 
C**    @(#)markh.f	1.1    3/10/88
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
C     CALL MARKH(HEIGHT)
C
C     THIS SUBROUTINE SPECIFIES THE SIZE OF A STANDARD DATA MARK.
C
C     THE PARAMETER AND ITS DEFINITION ARE LISTED BELOW:
C        HEIGHT - HEIGHT OF A STANDARD DATA MARK IN INCHES (OR
C                 MM).
C
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
C
      HMARK = HEIGHT * FACTOR / 8.0
      RETURN
C
      END
