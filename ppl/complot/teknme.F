	SUBROUTINE TEKNME(NME)
C** 
C**    @(#)teknme.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
	CHARACTER NME*(*)
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
	IF(TEKOPN)CLOSE(LUTK)
	TEKOPN=.FALSE.
	FILET=NME
	RETURN
	END
