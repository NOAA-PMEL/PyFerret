	SUBROUTINE SCINOT(X,Y,DEG,CSZE,NCHR,STR,ITYPE,WID,XF,YF)
C** 
C**    @(#)scinot.f	1.2    6/13/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
	CHARACTER STR*(*),ITMP*20,ITMP2*10,SUB*20,IC*1
	character ESTR*1,MESTR*2,lestr*1,lmestr*2
	data lestr,lmestr/'e','-e'/
	DATA ESTR,MESTR/'E','-E'/
c
	IF(NCHR.GT.20)RETURN
	ITMP=' '
	ITMP2=' '
	SUB=' '
	NS=0
	L=0
	J=1
	NE=0
	DO 10 NM=1,20
	IF(NM.GT.NCHR)GOTO 100
	IC=STR(NM:NM)
	IF(IC.EQ.' '.AND.J.EQ.1)GOTO 10
	IF(IC.EQ.estr.or.ic.eq.lestr.OR.IC.EQ.' ')GOTO 100
	IF(IC.EQ.'+'.OR.IC.EQ.'-')THEN
		IF(J.NE.1)THEN
			GOTO 100
		ELSE
			GOTO 20
		ENDIF
	ENDIF
	IF((IC.LT.'0'.OR.IC.GT.'9').AND.IC.NE.'.')THEN
		SUB=STR(NM:)
		NC=J-1
		NS=LNBLK(SUB,20)
		GOTO 1000
	ENDIF
20	ITMP(J:J)=IC
	J=J+1
10	CONTINUE
100	IF(IC.EQ.' '.OR.NM.GT.NCHR)GOTO 200
	NE=2
	IF(STR(1:1).NE.estr.AND.STR(1:2).NE.mestr
     *	.and.str(1:1).ne.lestr.and.str(1:2).ne.lmestr)THEN
	ITMP(J:J+1)=' x'
	J=J+2
	ENDIF
200	NC=J-1
	IF(NE.EQ.0)GOTO 1000
	IF(IC.EQ.estr.or.ic.eq.lestr)NM=NM+1
	IS=0
	DO 60 K=NM,20
	IF(K.GT.NCHR)GOTO 300
	IC=STR(K:K)
	IF(IC.EQ.' ')GOTO 60
	IF(IC.EQ.'+')GOTO 60
	IF(IC.EQ.'0'.AND.L+IS.EQ.0)GOTO 60
	IF((IC.LT.'0'.OR.IC.GT.'9').AND.IC.NE.'-')THEN
		SUB=STR(K:)
		NS=LNBLK(SUB,20)
		GOTO 300
	ENDIF
	IF(IC.EQ.'-')IS=-1
	L=L+1
	ITMP2(L:L)=IC
60	CONTINUE
300	IF(L.NE.0)GOTO 1000
	L=1
	ITMP2='0'
1000  C=COS(DEG*1.7453292E-2) 
      S=SIN(DEG*1.7453292E-2) 
      WID2=SYMWID(CSZE,NE,'10') 
      WID1=SYMWID(CSZE,NC,ITMP)+WID2 
      WID3=WID1+SYMWID(CSZE*.75,L,ITMP2) 
      WID=WID3+SYMWID(CSZE,NS,SUB)
      IF(ITYPE.EQ.3)RETURN 
      GO TO (1,2,3)ITYPE+1 
1     XL=X-WID*C*.5/XF+CSZE*.5*S/XF 
      YL=Y-WID*S*.5/YF-CSZE*.5*C/YF 
      GOTO 400 
2     XL=X-WID*C/XF+CSZE*.5*S/XF 
      YL=Y-WID*S/YF-CSZE*.5*C/YF 
      GOTO 400 
3     XL=X+CSZE*.5*S/XF 
      YL=Y-CSZE*.5*C/YF 
400   CALL SYMBEL(XL,YL,DEG,CSZE,NC,ITMP) 
	IF(NE.NE.0)THEN
      CALL SYMBEL(XL+(WID1-WID2)*C/XF,YL+(WID1-WID2)*S/YF, 
     *DEG,CSZE,NE,'10') 
      XLT=XL+WID1*C/XF-CSZE*.625*S/XF 
      YLT=YL+WID1*S/YF+CSZE*.625*C/YF 
      CALL SYMBEL(XLT,YLT,DEG,CSZE*.75,L,ITMP2) 
	ENDIF
	XL=XL+WID3*C/XF
	YL=YL+WID3*S/YF
	CALL SYMBEL(XL,YL,DEG,CSZE,NS,SUB)
      RETURN 
      END 
