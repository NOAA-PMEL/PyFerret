      SUBROUTINE DASHSZ(DN1,UP1,DN2,UP2)
C** 
C**    @(#)dashsz.f	1.1    3/10/88
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
C     CALL DASHSZ(DN1,UP1,DN2,UP2)
C
C     THIS SUBROUTINE SUPPLIES NEW VALUES FOR THE PARAMETERS
C     WHICH SPECIFY THE LENGTHS OF THE INDIVIDUAL SEGMENTS OF
C     A DASHED LINE.
C
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
C
C     SAVE THE NEW PARAMETERS IN AN ARRAY.
C
      DSEG(1) = FACTOR * DN1
      DSEG(2) = FACTOR * UP1
      DSEG(3) = FACTOR * DN2
      DSEG(4) = FACTOR * UP2
C
C     REINITIALIZE THE SEGMENT COUNTER.
C
      IDSEG = 1
      SEG = DSEG(1)
      RETURN
C
      END
