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
*
*	Mod 1.9.90 J Davison to reduce the maximum #pts in any one call to 
*	polyline to 100 as GKS 4.0 under VMS 5.2, i.e., DECWindows, blows up
*	with too many points
*
C
C	CALL GKSMV
C
C	This routine creates a binary file that can be used to
c	construct device dependent plotting code
c
c
#ifdef unix
	include 'tmap_pplv11inc/pltcom_dat.decl'
	include 'pplv11inc/PLTCOM.DAT'
	include 'tmap_pplv11inc/gkscm1_inc.decl'
	include 'pplv11inc/GKSCM1.INC'
#else
	INCLUDE 'tmap_pplv11inc:tmap_PLTCOM.DAT'
	INCLUDE 'tmap_pplv11inc:tmap_GKSCM1.INC'
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
*jd*	SUBROUTINE GKPLOT(XN,YN,IP,II) Number of args reduced to 3 for MAC 7.92
	SUBROUTINE GKPLOT(XN,YN,IP)
#ifdef core
        if(ip.ne.0)then
            call lineabs2(xn,yn)
        else
            call moveabs2(xn,yn)
        endif
#else
*	*jd* 1.9.90  Max was 10,000
	integer	max
	parameter (max=100)
*	*jd* 1.9.90

	DIMENSION X(max),Y(max)

	DATA IPT/0/

	IF(IP.EQ.0)THEN
	    CALL DOIT(IPT,X,Y)
	    IPT=1
	ELSE
	    IPT=IPT+1
	    IF(IPT.GT.max)THEN
		CALL DOIT(max,X,Y)
		IPT=2
		X(1)=X(max)
		Y(1)=Y(max)
	    ENDIF
	ENDIF
	X(IPT)=XN
	Y(IPT)=YN
#endif
	RETURN
	ENTRY PPLGFLUSH
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
