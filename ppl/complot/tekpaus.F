      SUBROUTINE TEKPAS
C** 
C**    @(#)tekpaus.f	1.1    3/10/88
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
C     CALL TEKPAS
C
C     THIS SUBROUTINE SETS ALPHAS MODE AND SUSPENDS EXECUTION
C     TO ALLOW THE USER TO INSPECT THE PLOTTED OUTPUT.  EXECU-
C     TION RESUMES WHEN A CHARACTER IS ENTERED FROM THE KEY-
C     BOARD.  THIS CHARACTER IS DISCARDED.  ON THE HP PLOTTER,
C     IN ADDITION TO SETTING ALPHAS MODE, TEKPAS TERMINATES
C     THE CURRENT PLOT AND SENDS FURTHER OUTPUT TO THE TERMINAL.
C     TEKPAS HAS NO EFFECT ON THE ZETA AND GERBER PLOTTERS.
C
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
C
C     CHECK THE PLOT TYPE.
C
      IF(PTYPE .EQ. 1 .OR. IABS(PTYPE) .GE. 2)THEN
C
C     PAUSE UNTIL THE USER RESPONDS.
C
        CALL PAGE
        CALL CHOUT(BEL,1)
        CALL CHOUT(BEL,1)
	CALL CHDMP
        CALL CHIN(IN,1,0)
      ENDIF
C
C     SET ALPHAS MODE.
C
      CALL ALPHAS
      RETURN
C
      END
