      SUBROUTINE SCALE3(XSCL,YSCL,ZSCL)
C** 
C**    @(#)scale3.f	1.1    3/10/88
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
C     CALL SCALE3(XSCL,YSCL,ZSCL)
C
C     THIS SUBROUTINE DEFINES SCALE FACTORS FOR SUBSEQUENT PLOT-
C     TING.  THE SCALE FACTORS DETERMINE THE RELATIVE SIZE OF
C     THE DISPLAY.
C
C     THE VARIOUS PARAMETERS AND THEIR DEFINITIONS ARE LISTED
C     BELOW:
C        XSCL   - SCALE FACTOR TO CONVERT FROM USER'S UNITS TO
C                 INCHES (OR MM) ON THE X-AXIS.
C        YSCL   - SCALE FACTOR TO CONVERT FROM USER'S UNITS TO
C                 INCHES (OR MM) ON THE Y-AXIS.
C        ZSCL   - SCALE FACTOR TO CONVERT FROM USER'S UNITS TO
C                 INCHES (OR MM) ON THE Z-AXIS.
C
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
C
C     CONVERT THE SCALE FACTORS TO INTERNAL INCREMENTS.
C
      XSCALE = FACTOR * XSCL
      YSCALE = FACTOR * YSCL
      ZSCALE = FACTOR * ZSCL
      RETURN
C
      END
