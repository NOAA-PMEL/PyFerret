	SUBROUTINE CROSS
C** 
C**    @(#)cross.F	1.1    3/10/88
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
	INCLUDE 'pplinc/PARAMPL5.DAT'
	INCLUDE 'pplinc/AXIS.INC'
	INCLUDE 'pplinc/LABCOM.INC'
	INCLUDE 'pplinc/LINES.INC'
#else
	INCLUDE 'pplinc:PARAMPL5.DAT'
	INCLUDE 'pplinc:AXIS.INC'
	INCLUDE 'pplinc:LABCOM.INC'
	INCLUDE 'pplinc:LINES.INC'
#endif
	IF(ICROSS.EQ.1)THEN
	    IF(XLO.LT.0..AND.XHI.GT.0.)THEN
		CALL PLOT(0.,YLO,0,0)
		CALL PLOT(0.,YHI,1,0)
	    ENDIF
	    IF(YLO.LT.0..AND.YHI.GT.0.)THEN
		CALL PLOT(XLO,0.,0,0)
		CALL PLOT(XHI,0.,1,0)
	    ENDIF
	ELSE IF(ICROSS.NE.0)THEN
	    IF(ICROSS.EQ.2.OR.ICROSS.EQ.4)THEN
		DO 90 I=1,LNUM
		DO 95 J=I+1,LNUM
		IF(YOFF(I).EQ.YOFF(J))GOTO90
95		CONTINUE
		CALL PLOT (XLO,YOFF(I),0,0)
		CALL PLOT (XHI,YOFF(I),1,0)
90		CONTINUE
	    ENDIF
	    IF(ICROSS.GE.3)THEN
		DO 100 I=1,LNUM
		DO 105 J=I+1,LNUM
		IF(XOFF(I).EQ.XOFF(J))GOTO 100
105		CONTINUE
		CALL PLOT(XOFF(I),YLO,0,0)
		CALL PLOT(XOFF(I),YHI,1,0)
100		CONTINUE
	    ENDIF
	ENDIF
	RETURN
	END
