	SUBROUTINE TEKLUN(LUN)
C** 
C**    @(#)teklun.f	1.1    3/10/88
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
C	CHANGE LUN OF TEK TERMINAL PLOT  DO NOT CHANGE WHILE PLOT
C	IS IN PROGRESS!!!!!
C	D. DENBO JUNE 1983
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
	LUTK=LUN
	RETURN
	END
