	SUBROUTINE LEVSRT
C** 
C**    @(#)levsrt.F	1.1    3/10/88
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
	INCLUDE 'pplinc/CONT.INC'
#else
        INCLUDE 'pplinc:PARAMPL5.DAT'
        INCLUDE 'pplinc:CONT.INC'
#endif
	LOGICAL BLNK
300	BLNK=.FALSE.
	DO 310 I=2,NLEV
	IF(ZLEV(I-1).GT.ZLEV(I))THEN
		BLNK=.TRUE.
		Z=ZLEV(I)
		J=LDIG(I)
		K=LWGT(I)
		JC=LCLR(I)
		ZLEV(I)=ZLEV(I-1)
		LDIG(I)=LDIG(I-1)
		LWGT(I)=LWGT(I-1)
		LCLR(I)=LCLR(I-1)
		ZLEV(I-1)=Z
		LDIG(I-1)=J
		LWGT(I-1)=K
		LCLR(I-1)=JC
	ENDIF
310	CONTINUE
	IF(BLNK)GOTO 300
	RETURN
	END
