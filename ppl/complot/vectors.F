      SUBROUTINE VECTRS
C** 
C**    @(#)vectors.f	1.1    3/10/88
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
C     CALL VECTRS
C
C     THIS SUBROUTINE CAUSES ALL SUBSEQUENT PLOTTING THAT IS
C     DONE WITH THE PEN DOWN TO BE IN THE FORM OF SMOOTH LINES
C     CONNECTING EACH OF THE SPECIFIED COORDINATES.
C
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
C
C     SET THE PLOTTING MODE.
C
      PMODE = VECTOR
      RETURN
C
      END
