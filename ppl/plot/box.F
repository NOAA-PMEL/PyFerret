      SUBROUTINE BOX 
C** 
C**    @(#)box.F	1.1    3/10/88
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
	INCLUDE 'pplinc/PLT.INC'
	INCLUDE 'pplinc/PEN.INC'
#else
	INCLUDE 'pplinc:PARAMPL5.DAT'
	INCLUDE 'pplinc:PLT.INC'
	INCLUDE 'pplinc:PEN.INC'
#endif
	CALL SCALE(1.,1.,0.,0.,0.,0.) 
	IF(IP(1).NE.1)RETURN
	CALL PLOT(.01,.01,0,0) 
	CALL PLOT(WIDTH-.02,.01,1,0) 
	CALL PLOT(WIDTH-.02,HEIGHT-.04,1,0) 
	CALL PLOT(.01,HEIGHT-.04,1,0) 
	CALL PLOT(.01,.01,1,0) 
	RETURN 
	END 
