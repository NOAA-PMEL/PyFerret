	SUBROUTINE DELLEV(J)
C** 
C**    @(#)dellev.F	1.1    3/10/88
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
	DO 520 K=J+1,NLEV
	ZLEV(K-1)=ZLEV(K)
	LDIG(K-1)=LDIG(K)
	LCLR(K-1)=LCLR(K)
520	LWGT(K-1)=LWGT(K)
	NLEV=NLEV-1
	RETURN
	END
