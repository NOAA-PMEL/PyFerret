	SUBROUTINE SCINOT_d(X,Y,DEG,CSZE,NCHR,STR,ITYPE,WID,XF,YF)
*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
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
C *acm* 11/05    Changes for deg-min-sec labelling. for these long
C                labels make the character size a bit smaller.
C 12/2022 *acm* Code cleanup: explicitly set type of arguments to SYMBEL

	REAL*8 x,y,xf,yf
	REAL   wid,wid1,wid2,wid3
        REAL charsize
	CHARACTER STR*(*),ITMP*20,ITMP2*10,SUB*20,IC*1
	character ESTR*1,MESTR*2,lestr*1,lmestr*2
	data lestr,lmestr/'e','-e'/
	DATA ESTR,MESTR/'E','-E'/
c
        charsize = CSZE
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
	IF((IC.LT.'0'.OR.IC.GT.'9').AND.IC.NE.'.'.AND.IC.NE.':')THEN
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
1000  IF (ns .GT. 6) charsize = 0.9* charsize
      C=COS(DEG*1.7453292E-2) 
      S=SIN(DEG*1.7453292E-2) 
      WID2=SYMWID(charsize,NE,'10') 
      WID1=SYMWID(charsize,NC,ITMP)+WID2 
      WID3=WID1+SYMWID(charsize*.75,L,ITMP2) 
      WID=WID3+SYMWID(charsize,NS,SUB)
      IF(ITYPE.EQ.3)RETURN 
      GO TO (1,2,3)ITYPE+1 
1     XL=X-WID*C*.5/XF+charsize*.5*S/XF 
      YL=Y-WID*S*.5/YF-charsize*.5*C/YF 
      GOTO 400 
2     XL=X-WID*C/XF+charsize*.5*S/XF 
      YL=Y-WID*S/YF-charsize*.5*C/YF 
      GOTO 400 
3     XL=X+charsize*.5*S/XF 
      YL=Y-charsize*.5*C/YF 
400   CALL SYMBEL(XL,YL,DEG,charsize,NC,ITMP) 
	IF(NE.NE.0)THEN
      CALL SYMBEL(SNGL(XL+(WID1-WID2)*C/XF),
     * SNGL(YL+(WID1-WID2)*S/YF), DEG,charsize,NE,'10') 
      XLT=XL+WID1*C/XF-charsize*.625*S/XF 
      YLT=YL+WID1*S/YF+charsize*.625*C/YF 
      CALL SYMBEL(XLT,YLT,DEG,charsize*.75,L,ITMP2) 
	ENDIF
	XL=XL+WID3*C/XF
	YL=YL+WID3*S/YF
	CALL SYMBEL(XL,YL,DEG,charsize,NS,SUB)
      RETURN 
      END 
