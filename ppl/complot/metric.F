      SUBROUTINE METRIC
C** 
C**    @(#)metric.f	1.1    3/10/88
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
C     CALL METRIC
C
C     THIS SUBROUTINE CHANGES AN INTERNAL CONVERSION FACTOR SO
C     THAT ALL COMPLOT SUBROUTINES WHICH ARE SUBSEQUENTLY CALLED
C     AND ORDINARILY EXPECT TO RECEIVE PARAMETERS EXPRESSED IN
C     INCHES WILL INSTEAD EXPECT TO RECEIVE THE PARAMETERS EX-
C     PRESSED IN MILLIMETERS.
C
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
C
      FACTOR = 39.37
      RETURN
C
      END
