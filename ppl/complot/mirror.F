      SUBROUTINE MIRROR(IMIRA,IMIRB)
C** 
C**    @(#)mirror.f	1.1    3/10/88
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
C     CALL MIRROR(IMIRA,IMIRB)
C
C     THIS SUBROUTINE SELECTS THE TYPE OF MIRROR IMAGE, IF ANY,
C     WHICH WILL BE DRAWN.
C
C     THE VARIOUS PARAMETERS AND THEIR DEFINITIONS ARE LISTED
C     BELOW:
C        IMIRA  - HORIZONTAL MIRROR-IMAGE CODE (0 = OFF, NON-
C                 ZERO = ON).
C        IMIRB  - VERTICAL MIRROR-IMAGE CODE (0 = OFF, NON-ZERO
C                 = ON).
C
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
C
C     SET THE MIRROR-IMAGE FLAGS.
C
      MIRAF = .FALSE.
      IF(IMIRA .NE. 0)MIRAF = .TRUE.
      MIRBF = .FALSE.
      IF(IMIRB .NE. 0)MIRBF = .TRUE.
      RETURN
C
      END
