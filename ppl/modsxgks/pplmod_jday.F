	INTEGER*4 FUNCTION JDAY(IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)
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
*	Mod J Davison 11.22.95 For 4 digit year

*       Year to be calendar year - 1900, eg, 1880 = -20, 2001 = 101
	IYCEN = IYR + (ICEN - 19)*100

	IF(IMON.GT.2)THEN
	   IM=IMON+1
	   IY=IYCEN
	ELSE
	   IM=IMON+13
	   IY=IYCEN-1
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
