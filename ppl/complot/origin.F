      SUBROUTINE ORIGIN(AORG,BORG)
C** 
C**    @(#)origin.f	1.1    3/10/88
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
C     CALL ORIGIN(AORG,BORG)
C
C     THIS SUBROUTINE DEFINES THE LOCATION OF THE ORIGIN OF THE
C     DISPLAY.
C
C     THE VARIOUS PARAMETERS AND THEIR DEFINITIONS ARE LISTED
C     BELOW:
C        AORG   - HORIZONTAL DISPLACEMENT OF THE ORIGIN IN IN-
C                 CHES (OR MM).
C        BORG   - VERTICAL DISPLACEMENT OF THE ORIGIN IN INCHES
C                 (OR MM).
C
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
C
C     CONVERT THE PARAMETERS TO INTERNAL INCREMENTS.
C
      AORIGN = FACTOR * AORG
      BORIGN = FACTOR * BORG
      RETURN
C
      END
