      SUBROUTINE PLOT3(X,Y,Z,IPEN,IMARK)
C** 
C**    @(#)plot3.f	1.1    3/10/88
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
C     CALL PLOT3(X,Y,Z,IPEN,IMARK)
C
C     THIS SUBROUTINE MOVES THE PEN FROM THE CURRENT POSITION TO
C     A NEW POINT.
C
C     THE VARIOUS PARAMETERS AND THEIR DEFINITIONS ARE LISTED
C     BELOW:
C        X,Y,Z  - COORDINATES OF THE NEW POINT IN USER'S UNITS.
C        IPEN   - PEN POSITION (0 = PEN UP, NON-ZERO = PEN DOWN).
C        IMARK  - DATA MARK CODE (0-88).
C
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
C
C     CHECK THE PLOTTING MODE TO INSURE THAT IT IS SET FOR GRAPH-
C     ICS.
C
      IF(PMODE .EQ. ALPHA)PMODE = VECTOR
C
C     SCALE THE COORDINATES OF THE NEW POINT.
C
      XTEMP = X * XSCALE
      YTEMP = Y * YSCALE
      ZTEMP = Z * ZSCALE
C
C     SET THE PEN POSITION.
C
      PENF = .FALSE.
      IF(IPEN .NE. 0)PENF = .TRUE.
C
C     PLOT TO THE POINT.
C
      CALL XYZPLT
C
C     DETERMINE WHETHER A DATA MARK SHOULD BE DRAWN.
C
      IF(IPEN .NE. 0)CALL MARK(IMARK)
      RETURN
C
      END
