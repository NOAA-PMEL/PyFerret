      SUBROUTINE HABIS(IPLOT,IPAUSE,IHDCPY,IN)
C** 
C**    @(#)habis.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
#ifdef unix
	INCLUDE 'pltinc/PLTCOM.DAT'
	INCLUDE 'cmdinc/SYSTEM.INC'
#else
	include '[-.complot]pltcom.dat'
	include 'cmdinc:system.inc'
	byte in
#endif
	CHARACTER INS*1
C
C     THIS ROUTINE IS OUR ALTERNATIVE TO PLOTEND.
C         IPLOT=XXXX0YYYY0Z
C             XXXX=BAUD
C             YYYY=TEK MODEL
C             Z=ITYPE (PLOT TYPE)
C         IPAUSE = 0 FOR NO PAUSE
C                = 1 FOR A  PAUSE
C         IHDCPY= 0 FOR NO HARDCOPY
C                = 1 FOR A HARD COPY
C
	CALL ALPHAS
	CALL PAGE
	IF(IHDCPY.EQ.1) CALL HDCOPY(IPLOT)
	IF(IPAUSE.EQ.1.AND.(PTYPE.EQ.1.OR.IABS(PTYPE).GE.2)) THEN
	    CALL CHOUT(BEL,1)
	    CALL CHOUT(BEL,1)
	    CALL CHDMP
	    IF(PTYPE.GE.3)THEN
#ifdef unix
		OPEN(99,FILE='/dev/tty',STATUS='UNKNOWN')
#else
		OPEN(99,FILE='TT:',STATUS='UNKNOWN')
#endif
		READ(99,'(A1)',END=100)INS
100		IN=ICHAR(INS)
	    ELSE
		CALL CHIN(IN,1,0)
	    ENDIF
	ENDIF
	call clsdev(termf,sover)
	RETURN
	END
