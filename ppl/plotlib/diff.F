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
	CHARACTER*10 IT1,IT2
	INTEGER*4 JDAY
	CALL TIME3(IT1,IY1,IM1,ID1,IH1,IMN1)
	CALL TIME3(IT2,IY2,IM2,ID2,IH2,IMN2)
#ifdef unix
	DIFF=FLOAT(JDAY(IY2,IM2,ID2,IH2,IMN2)-
     * JDAY(IY1,IM1,ID1,IH1,IMN1))
#else
	DIFF=FLOATJ(JDAY(IY2,IM2,ID2,IH2,IMN2)-
     * JDAY(IY1,IM1,ID1,IH1,IMN1))
#endif
	RETURN
	END
