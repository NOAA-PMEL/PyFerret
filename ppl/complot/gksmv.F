	SUBROUTINE GKSMV
C** 
C**    @(#)gksmv.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C
C	CALL GKSMV
C
C	This routine creates a binary file that can be used to
c	construct device dependent plotting code
c
c
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
	include 'pplinc/GKSCM1.INC'
#else
	include 'pplinc:PLTCOM.DAT'
	include 'pplinc:GKSCM1.INC'
#endif
	IF(.NOT.GKSOPN)THEN
#ifdef core
	    call crinit
#else
	    call gkinit
C
	    CALL GCLRWK(WSID,0)
#endif
	    GKSOPN=.TRUE.
	    LPEN=.NOT.PEN
	ENDIF
	IF(PEN)THEN
	CALL GKPLOT(A,B,1)
	ELSE
	CALL GKPLOT(A,B,0)
	ENDIF
	RETURN
	END
*	SUBROUTINE GKPLOT(XN,YN,IP,II) *jd* 7.92 Param list reduced for MAC
	SUBROUTINE GKPLOT(XN,YN,IP)
#ifdef core
	if(ip.ne.0)then
	    call lineabs2(xn,yn)
	else
	    call moveabs2(xn,yn)
	endif
#else
	DIMENSION X(10000),Y(10000)
	DATA IPT/0/
	IF(IP.EQ.0)THEN
	    CALL DOIT(IPT,X,Y)
	    IPT=1
	ELSE
	    IPT=IPT+1

	    IF(IPT.GT.10000)THEN
		CALL DOIT(10000,X,Y)
		IPT=2
		X(1)=X(10000)
		Y(1)=Y(10000)
	    ENDIF
	ENDIF
	X(IPT)=XN
	Y(IPT)=YN
#endif
	RETURN
	ENTRY GFLUSH
#ifdef core
	return
	end
#else
	CALL DOIT(IPT,X,Y)
	IPT=0
	RETURN
	END
	SUBROUTINE DOIT(IPT,X,Y)
	DIMENSION X(*),Y(*)
	IF(IPT.LE.0)RETURN
	IF(IPT.EQ.1)THEN
	    X(2)=X(1)
	    Y(2)=Y(1)
	    IPT=2
	ENDIF
	CALL GPL(IPT,X,Y)
	RETURN
	END
#endif

