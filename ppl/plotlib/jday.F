	INTEGER*4 FUNCTION JDAY(IYR,IMON,IDAY,IHR,IMIN)
C** 
C**    @(#)jday.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
	IF(IMON.GT.2)THEN
	IM=IMON+1
	IY=IYR
	ELSE
	IM=IMON+13
	IY=IYR-1
	ENDIF
#ifdef unix
	JDAY=(INT(365.25*IY)+INT(30.6001*IM)+IDAY)*1440+
     *	IHR*60+IMIN
#else
	JDAY=(JINT(365.25*IY)+JINT(30.6001*IM)+IDAY)*1440+
     *	IHR*60+IMIN
#endif
	RETURN
	END
