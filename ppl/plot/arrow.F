	SUBROUTINE ARROW(XL,YL,XL2,YL2,IAT,SHFT,XMN,CS,SN,XF,YF)
C** 
C**    @(#)arrow.F	1.1    3/10/88
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
	INCLUDE 'pplinc/LABCOM.INC'
#else
	INCLUDE 'pplinc:LABCOM.INC'
#endif
	IF(IAT.EQ.0)RETURN
	IF(XL+SHFT*CS/XF.GE.XL2)THEN
	    AX=-HLABS*.5
	    AY=HLABS*.5
	ELSE IF(XL+(XMN+SHFT*CS)/XF.LE.XL2)THEN
	    AX=(HLABS*.5+XMN)
	    AY=HLABS*.5
	ELSE IF(YL+SHFT*SN/YF.GT.YL2)THEN
	    AY=-HLABS*.5
	    AX=XMN*.5
	ELSE
	    AY=HLABS*1.5
	    AX=XMN*.5
	ENDIF
	XLT=XL+((SHFT+AX)*CS-AY*SN)/XF
	YLT=YL+((SHFT+AX)*SN+AY*CS)/YF
	IF(IAT.EQ.2)THEN
	    HS=HLABS*0.5
	    DNM=SQRT(((XL2-XLT)*XF)**2+((YL2-YLT)*YF)**2)
	    CCS=(XL2-XLT)*XF/DNM
	    SSN=(YL2-YLT)*YF/DNM
	    XN=XL2+(-CCS-.5*SSN)*HS/XF
	    YN=YL2+(-SSN+.5*CCS)*HS/YF
	    CALL PLOT(XN,YN,0,0)
	    CALL PLOT(XL2,YL2,1,0)
	    XN=XL2+(-CCS+.5*SSN)*HS/XF
	    YN=YL2+(-SSN-.5*CCS)*HS/YF
	    CALL PLOT(XN,YN,1,0)
	ENDIF
	CALL PLOT(XL2,YL2,0,0)
	CALL PLOT(XLT,YLT,1,0)
	RETURN
	END
