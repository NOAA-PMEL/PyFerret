	SUBROUTINE LEVEL
C** 
C**    @(#)level.F	1.1    3/10/88
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
	INCLUDE 'pplinc/CMRD.INC'
	INCLUDE 'pplinc/CONT.INC'
#else
        INCLUDE 'pplinc:PARAMPL5.DAT'
        INCLUDE 'pplinc:CMRD.INC'
        INCLUDE 'pplinc:CONT.INC'
#endif
	DO 2601 I=1,M 
	IF(IFLG(I).NE.0)ZLEV(I)=P(I) 
2601	CONTINUE 
	NLEV=M 
	IF(M.LE.1)THEN 
	    NLEV=IFIX(P(1)) 
	    IF(M.EQ.0)NLEV=10 
	    IAUTOC=1 
	    GOTO 10 
	ELSE 
	    IAUTOC=0 
	    JJ=NSIGC-IFIX(ALOG10(ABS(ZLEV(1)-ZLEV(NLEV)))+.5) 
	    IF(JJ.LE.0)JJ=-1 
	    DO 2602 I=1,NLEV 
	    LCLR(I)=0
	    IF(ZLEV(I).EQ.0.)THEN 
	      LDIG(I)=-1 
	    ELSE 
		LDIG(I)=JJ
	    END IF
2602	    CONTINUE
	END IF 
	IFLG(1)=0 
	DO 2603 I=1,NLEV 
	LWGT(I)=2 
	IF(ABS(ZLEV(I)).LT.ABS(ZLEV(1)-ZLEV(2))*1.E-5)GOTO 2603 
	IF(ZLEV(I).GT.0.)THEN 
	    LWGT(I)=1 
	ELSE 
	    LWGT(I)=3 
	END IF 
2603	CONTINUE 
10	RETURN
	END
