	SUBROUTINE SAVEZG(LUN)
C** 
C**    @(#)savezg.F	1.1    3/10/88
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
	INCLUDE 'pplinc/ZGRIDD.INC'
	INCLUDE 'pplinc/ZGRID.INC'
#else
        INCLUDE 'pplinc:ZGRIDD.INC'
        INCLUDE 'pplinc:ZGRID.INC'
#endif
	WRITE(LUN)(XX(I),YY(I),ZZ(I),I=1,IZLEN)
	RETURN
	ENTRY GETZG(LUN)
	READ(LUN)(XX(I),YY(I),ZZ(I),I=1,IZLEN)
	RETURN
	END
