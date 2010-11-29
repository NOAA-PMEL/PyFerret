c
c	gkscm1
C** 
C**    @(#)GKSCM1.INC	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
*	JD Corrected for v1.1 typo SH found ie, YKYMX --> WKYMX

c
	COMMON/GKSCM1/WSID,CONID,WSTYPE,ERRFIL,CHDEV,
     *	CHPET,LCDEV,LCPET,WKSAUT,WKXMN,WKXMX,WKYMN,WKYMX
	INTEGER WSID,CONID,WSTYPE,ERRFIL,CHDEV,CHPET,
     *	LCDEV,LCPET
	REAL WKXMN,WKXMX,WKYMN,WKYMX
	LOGICAL WKSAUT
c
c	GKS common block for workstation id, type and device
c	and prompt-echo-type
c
c	wsid	workstation id
c	conid	connection id
c	wstype	workstation type
c	errfil	logical unit for error file
c	chdev	choice device for wsid
c	chpet	choice prompt echo type for wsid
c	lcdev	locator device for wsid
c	lcpet	locator prompt echo type for wsid
c	wksaut	if true use the default workstation viewport
c		if false use wkxmn ... wkymx for workstation
c		the viewport
c	wkxmn	minimum x value for viewport (meters)
c	wkxmx	maximum x value for viewport
c	wkymn	minimum y value for viewport
c	wkymx	maximum y value for viewport
c
c*******************************************************************
