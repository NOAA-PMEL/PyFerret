      SUBROUTINE OBLIQ3(DEGRES)
C** 
C**    @(#)obliq3.f	1.1    3/10/88
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
C     CALL OBLIQ3(DEGRES)
C
C     THIS SUBROUTINE ESTABLISHES CONTROL INFORMATION FOR OBLIQUE
C     THREE-DIMENSIONAL PROJECTIONS WHERE THE PLANE OF PROJECTION
C     IS PARALLEL TO THE XY-PLANE.
C
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
C
      IF(DEGRES .NE. 0.0)THEN
C
C     SET THE VIEW FLAG.
C
        VIEWF = .TRUE.
C
C     CONVERT THE ANGLE OF PROJECTION TO RADIANS.
C
        TEMP = 0.01745329251994 * DEGRES
C
C     COMPUTE THE ELEMENTS OF THE TRANSFORMATION MATRIX.
C
        TAX = 1.0
        TAY = 0.0
        TAZ = -COS(TEMP)
        TBX = 0.0
        TBY = 1.0
        TBZ = -SIN(TEMP)
        TCX = 0.0
        TCY = 0.0
        TCZ = 0.0
        VDIST = 0.0
      ELSE
C
C     RESET THE VIEW FLAG.
C
        VIEWF = .FALSE.
      ENDIF
      RETURN
C
      END
