	SUBROUTINE PLTNME(NME)
C** 
C**    @(#)pltnme.f	1.1    3/10/88
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
	IF(PLTFLG)THEN
	    PEN=.FALSE.
	    IF(PTYPE.LT.0)THEN
		A=0.
		B=0.
		CALL HPABMV
		CALL ZBUFFT('SP0;',4)
		OLDHTA=0
		CALL HFLUSH
	    ELSE IF(PTYPE.EQ.0.OR.PTYPE.EQ.2.OR.PTYPE.EQ.4)THEN
		CALL ZABMV
		CALL BINFSH
	    ENDIF
	    PLTFLG=.FALSE.
	ENDIF
	FILEZ=NME
	RETURN
	END
