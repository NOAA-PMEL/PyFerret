      SUBROUTINE TKTYPE(ICODE)
C** 
C**    @(#)tktype.f	1.1    3/10/88
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
C     CALL TKTYPE(ICODE)
C
C     COMPLOT ASSUMES THAT A TEKTERMINAL IS A 4010.  IF A DIF-
C     FERENT TYPE OF TEKTRONIX GRAPHICS TERMINAL IS USED INSTEAD,
C     THIS SUBROUTINE MUST BE CALLED TO MAKE THE APPROPRIATE
C     CHANGES TO CONTROL INFORMATION.
C
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
C
C     CHECK THE TEKTERMINAL MODEL.
C
	TTYPE=ICODE
C
      IF(ICODE .EQ. 4010)GO TO 40
      IF(ICODE .EQ. 4014)GO TO 50
      IF(ICODE .EQ. 4107)GO TO 50
      IF(ICODE .EQ. 4115)GO TO 50
      IF(ICODE .EQ. 4051)GO TO 70
      IF(ICODE .EQ. 4052)GO TO 70
      IF(ICODE .EQ. 4662)GO TO 170
      RETURN
C
C     CHANGE CONTROL INFORMATION.
C
   40 TWIDTH = 7.5
      THIGHT = 5.625
      TSCLA = 1023.0
      TSCLB = 779.0
      GO TO 250
C
C     CHANGE CONTROL INFORMATION.
C
   50 TWIDTH = 7.8448
      THIGHT = 6.0
      TSCLA = 4095.0
      TSCLB = 3132.0
      GO TO 250
C
C     CHANGE CONTROL INFORMATION.
C
   70 TWIDTH = 7.875
      THIGHT = 6.0
      TSCLA = 1023.0
      TSCLB = 779.0
      GO TO 250
  170 IDTK = 67
      TTYPE = -4662
C
C     CHANGE CONTROL INFORMATION.
C
      TWIDTH = 15.0
      THIGHT = 10.0
      TSCLA = 4095.0
      TSCLB = 2731.0
  250 FACTOR = 1000.0
C
C     DEFINE THE DEFAULT DIMENSIONS OF THE PLOTTING AREA.
C
c
c	keep from resizing the display
c
c      CALL SIZE(TWIDTH,THIGHT)
      RETURN
C
      END
