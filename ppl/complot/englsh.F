      SUBROUTINE ENGLSH
C** 
C**    @(#)englsh.f	1.1    3/10/88
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
C     CALL ENGLSH
C
C     THIS SUBROUTINE REVERSES THE EFFECT OF SUBROUTINE METRIC
C     BY RESETTING AN INTERNAL CONVERSION FACTOR SO THAT ALL
C     COMPLOT SUBROUTINES WHICH ARE SUBSEQUENTLY CALLED AND OR-
C     DINARILY EXPECT TO RECEIVE PARAMETERS EXPRESSED IN INCHES
C     WILL RESUME DOING SO.
C
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
C
      FACTOR = 1000.0
      RETURN
C
      END
