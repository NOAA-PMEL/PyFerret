	FUNCTION DIFF(IT1,IT2)
C** 
C**    @(#)diff.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
*	Mod J Davison 11.20.95 for 4 digit year and seconds

	CHARACTER*(*) IT1,IT2
	INTEGER*4 JDAY
	REAL*8 DATE2, DATE1, WHOI2BC, RDATE
	
	RDATE = WHOI2BC(IT2) - WHOI2BC(IT1)
	DIFF  = RDATE

*       Original code
*	CALL TIME3(IT1,IY1,IM1,ID1,IH1,IMN1,ISC1,ICN1)
*	CALL TIME3(IT2,IY2,IM2,ID2,IH2,IMN2,ISC2,ICN2)
*#ifdef unix
*	DIFF=FLOAT(JDAY(IY2,IM2,ID2,IH2,IMN2,ISC2,ICN2)-
*     * JDAY(IY1,IM1,ID1,IH1,IMN1,ISC1,ICN1))
*#else
*	DIFF=FLOATJ(JDAY(IY2,IM2,ID2,IH2,IMN2,ISC2,ICN2)-
*     * JDAY(IY1,IM1,ID1,IH1,IMN1,ISC1,ICN1))
*#endif

	RETURN
	END
