	SUBROUTINE LEVSET(IST,VAL,IF,LWGH,NL)
C** 
C**    @(#)levset.F	1.1    3/10/88
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
	INCLUDE 'pplinc/CMRDL.INC'
#else
	INCLUDE 'pplinc:CMRDL.INC'
#endif
	DIMENSION VAL(4),IF(4)
	LWGH=0
	IF(INDEX(LABEL(:IST),'DEL').NE.0)LWGT=5
	IF(INDEX(LABEL(:IST),'PEN').NE.0)LWGH=4
	IF(INDEX(LABEL(:IST),'DASH').NE.0)LWGH=3
	IF(INDEX(LABEL(:IST),'DARK').NE.0)LWGH=2
	IF(INDEX(LABEL(:IST),'LINE').NE.0)LWGH=1
	IF(IF(2).EQ.0)THEN
		NL=1
	ELSE
		IF(IF(3).EQ.0)THEN
			VAL(3)=VAL(2)-VAL(1)
			NL=2
		ELSE
			IF(VAL(3).NE.0.)THEN
			NL=INT((VAL(2)-VAL(1))/VAL(3)+0.00001)+1
			ELSE
			NL=2
			ENDIF
		ENDIF
	ENDIF
	RETURN
	END
