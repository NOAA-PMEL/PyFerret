	SUBROUTINE PLOT(X,Y,IPEN,IDUM)
C** 
C**    @(#)plotnm.F	1.1    3/10/88
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
C	PLOT WITH MARK REMOVED
C
#ifdef unix
	INCLUDE 'pltinc/PLTCOM.DAT'
#else
	include 'pltinc:pltcom.dat'
#endif
	IF(PMODE.EQ.ALPHA)PMODE=VECTOR
	XTEMP=X*XSCALE
	YTEMP=Y*YSCALE
	PENF=.FALSE.
	IF(IPEN.NE.0)PENF=.TRUE.
	CALL XYZPLT
	RETURN
	END
