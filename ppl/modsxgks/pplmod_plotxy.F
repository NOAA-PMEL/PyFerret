	SUBROUTINE PLOTXY(X,NXS,XF,YF)
C** 
C**    @(#)plotxy.F	1.1    3/10/88
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
C  ARRAY X HOLDS X AND Y'S FOR ALL LINES.  DIMENSIONED FOR NXS.
C  ORDER AS FOLLOWS:  ALL X'S FOR LINE1, ALL X'S FOR LINE2,...
C                     STARTING AT X(NXS/2+1), ALL Y'S FOR LINE1,
C                     ALL Y'S FOR LINE2...
C  XF AND YF ARE FACTORS TO CONVERT TO INCHES FROM USER UNITS
C
*	Modified *jd* J Davison JISAO/PMEL/TMAP 3.22.89
*	To exlude Denbo line dashing in favor of GKS line bundles 
*	Mod *jd* 8.3.92 to add interrupts 

#ifdef unix
        include         'tmap_pplv11inc/xinterrupt.cmn' ! 1/89 *sh*
        include         'tmap_pplv11inc/ppl_in_ferret.cmn'
#else
        INCLUDE         'TMAP_PPLV11INC:XINTERRUPT.CMN' ! 1/89 *sh*
        INCLUDE         'TMAP_PPLV11INC:PPL_IN_FERRET.CMN'
#endif

	DIMENSION X(NXS)

#ifdef unix
        include 'tmap_pplv11inc/parampl5_dat.decl'
        include 'pplv11inc/PARAMPL5.DAT'
        include 'tmap_pplv11inc/axis_inc.decl'
        include 'pplv11inc/AXIS.INC'
        include 'tmap_pplv11inc/lines_inc.decl'
        include 'pplv11inc/LINES.INC'
        include 'tmap_pplv11inc/pen_inc.decl'
        include 'pplv11inc/PEN.INC'
        include 'tmap_pplv11inc/pltl_inc.decl'
        include 'pplv11inc/PLTL.INC'
        include 'tmap_pplv11inc/hd_inc.decl'
        include 'pplv11inc/HD.INC'
        include 'tmap_pplv11inc/miss_inc.decl'
        include 'pplv11inc/MISS.INC'
        include 'tmap_pplv11inc/taxis_inc.decl'
        include 'pplv11inc/TAXIS.INC'
        include 'tmap_pplv11inc/axisl_inc.decl'
        include 'pplv11inc/AXISL.INC'
        include 'tmap_pplv11inc/cmrd_inc.decl'
        include 'pplv11inc/CMRD.INC'
        include 'tmap_pplv11inc/gkscm2.cmn' ! jd 3.22.89
#else
        INCLUDE 'tmap_pplv11inc:tmap_PARAMPL5.DAT'
        INCLUDE 'tmap_pplv11inc:tmap_AXIS.INC'
        INCLUDE 'tmap_pplv11inc:tmap_LINES.INC'
        INCLUDE 'tmap_pplv11inc:tmap_PEN.INC'
        INCLUDE 'tmap_pplv11inc:tmap_PLTL.INC'
        INCLUDE 'tmap_pplv11inc:tmap_HD.INC'
        INCLUDE 'tmap_pplv11inc:tmap_MISS.INC'
        INCLUDE 'tmap_pplv11inc:tmap_TAXIS.INC'
        INCLUDE 'tmap_pplv11inc:tmap_AXISL.INC'
        INCLUDE 'tmap_pplv11inc:tmap_CMRD.INC'
        INCLUDE 'tmap_pplv11inc:tmap_gkscm2.inc' ! JD 3.22.89
#endif

	REAL MAXVEL

	IPV=0
C
C  JTYPE IS DATA TYPE (0 FOR 2D, 1 FOR 1D)
C
	IF(JTYPE.NE.1)RETURN
1000	IBASE=0
	IY=NXS/2
	CS=COS(VANG*.01745329252)
	SN=SIN(VANG*.01745329252)
	ZT=YF/XF
	ICNT=1
	IF(M.GE.2.AND.IFLG(2).NE.0)ICNT=P(2)

	IF(MAKEP.EQ.'PLOTUV')THEN
	  IBASE=IY
	  IBASE1=0
	  IBASE2=LLENG(1)+IY
	  DO 90 I=1,LNUM,2
	    CALL VECTRS
	    CALL COLOR(IPEN(I))
	    DO 95 J=1,LLENG(I),ICNT
	      IF(XLE.AND.X(IBASE+J).LE.CMXLE)GOTO 95
	      IF(XEQ.AND.X(IBASE+J).EQ.CMXEQ)GOTO 95
	      IF(XGE.AND.X(IBASE+J).GE.CMXGE)GOTO 95
	      IF(YLE.AND.X(IBASE2+J).LE.CMYLE)GOTO 95
	      IF(YEQ.AND.X(IBASE2+J).EQ.CMYEQ)GOTO 95
	      IF(YGE.AND.X(IBASE2+J).GE.CMYGE)GOTO 95
	      YT= YFCT(I)*X(IBASE2+J)*CS + XFCT(I)*X(IBASE+J)*SN
	      XT= (XFCT(I)*X(IBASE+J)*CS - YFCT(I)*X(IBASE2+J)*SN)*ZT
	      CALL PLOT(X(IBASE1+J)+XOFF(I),YOFF(I),0,0)
	      CALL PLOT(X(IBASE1+J)+XOFF(I)+XT,YOFF(I)+YT,1,0)
	      if (interrupted) goto 51 ! *jd* 8.3.92
95	    CONTINUE
	    IBASE=IBASE2+LLENG(I+1)
	    IBASE1=IBASE1+LLENG(I+1)+LLENG(I)
90	  IBASE2=IBASE+LLENG(I+2)
	  GOTO 51
	END IF

	DO 50 I=1,LNUM
	  CALL MARKH(HMARK(I))
	  CALL COLOR(IPEN(I))

C     SET PLOT MODE
	  IF(IPV.EQ.1)THEN
C     DRAW VECTORS
	    CALL VECTRS
	    DO 80 J=1,LLENG(I),ICNT
	      IF(XLE.AND.X(IBASE+J).LE.CMXLE)GOTO 80
	      IF(XEQ.AND.X(IBASE+J).EQ.CMXEQ)GOTO 80
	      IF(XGE.AND.X(IBASE+J).GE.CMXGE)GOTO 80
	      IF(YLE.AND.X(IY+IBASE+J).LE.CMYLE)GOTO 80
	      IF(YEQ.AND.X(IY+IBASE+J).EQ.CMYEQ)GOTO 80
	      IF(YGE.AND.X(IY+IBASE+J).GE.CMYGE)GOTO 80
	      YT= YFCT(I)*X(IY+IBASE+J)*CS + XFCT(I)*X(IBASE+J)*SN
	      XT=(XFCT(I)*X(IBASE+J)*CS - YFCT(I)*X(IY+IBASE+J)*SN)*ZT
	      CALL PLOT(FLOAT(J)+XOFF(I),YOFF(I),0,0)
	      CALL PLOT(FLOAT(J)+XOFF(I)+XT,YOFF(I)+YT,1,0)
80          CONTINUE

	  ELSE IF(IPV.EQ.2) THEN
	    IF( m .ge. 1 .and. iflg(1) .ne. 0)then
	       if( p(1) .gt. 0 )then
		   rlenfact = p(1) * abs(xf)
	       else
		   rlenfact = -p(1) * abs(yf)
	       endif
	    else
	       rlenfact = min( abs(xf), abs(yf))
	    endif
	    jcnt = 1
	    if( m .ge. 2 .and. iflg(2) .ne. 0)jcnt = p(2)
	    IF ( i .GT. 1) GOTO 50
	    ibaseu = 0
	    ibasev = iy
	    ibasex = lleng(1)
	    ibasey = iy + lleng(1)
	    CALL COLOR(IPEN(1))
	    maxvel = 0.0
	    DO 100 j = 1,lleng(1),jcnt
	      dist=sqrt(x(ibasev+j)**2+x(ibaseu+j)**2)
	      if(dist.gt.maxvel)maxvel=dist
	      ytail = x(ibasey+j)
	      yhead = ytail + rlenfact*x(ibasev+j)/yf
	      xtail = x(ibasex+j)
	      xhead = xtail + rlenfact*x(ibaseu+j)/xf
	      CALL DRAWVEC(xtail,ytail,xhead,yhead,rlenfact,0,xf,yf)
100	    CONTINUE
	    CALL DRAWVEC(maxvel,ytail,xhead,yhead,rlenfact,1,xf,yf)
	  ELSE
	    IF(ITYPE(I).GE.4 .and. .not.gks_open)THEN ! JD 3.22.89
	      CALL DASHSZ(DN1(I),UP1(I),DN2(I),UP2(I))
	      CALL DASHES
	    ELSE
	      CALL VECTRS
	    END IF
	    IF(ITYPE(I).EQ.3)CALL POINTS
	    IF(.NOT.(XLE.OR.XEQ.OR.XGE.OR.YLE.OR.YEQ.OR.YGE))THEN
C     DO NOT CHECK POINTS FOR MISSING DATA
	      CALL TRANS(I,X(IBASE+1),X(IY+IBASE+1),XT,YT)
	      CALL PLOT(XT,YT,0,0)
	      IF(ITYPE(I).NE.0.AND.ITYPE(I).NE.4)CALL MARK(IMARK(I))
	      IM=0
	      IF(ITYPE(I).EQ.1.OR.ITYPE(I).EQ.3)IM=IMARK(I)
	      DO 60 J=IBASE+1,IBASE+LLENG(I)
	        CALL TRANS(I,X(J),X(IY+J),XT,YT)
	        CALL PLOT(XT,YT,1,0)
60            CALL MARK(IM)
	    ELSE
C     CHECK POINTS FOR OUT OF RANGE   (CM)
	      DO 70 J=IBASE+1,IBASE+LLENG(I)
		IF(XLE.AND.X(J).LE.CMXLE)GOTO 70
		IF(XEQ.AND.X(J).EQ.CMXEQ)GOTO 70
		IF(XGE.AND.X(J).GE.CMXGE)GOTO 70
		IF(YLE.AND.X(IY+J).LE.CMYLE)GOTO 70
		IF(YEQ.AND.X(IY+J).EQ.CMYEQ)GOTO 70
		IF(YGE.AND.X(IY+J).GE.CMYGE)GOTO 70
	        CALL TRANS(I,X(J),X(IY+J),XT,YT)
	        CALL PLOT(XT,YT,0,0)
	        IF(ITYPE(I).NE.0.AND.ITYPE(I).NE.4)CALL MARK(IMARK(I))
 	        IM=0
	        IF(ITYPE(I).EQ.1.OR.ITYPE(I).EQ.3)IM=IMARK(I)
	        GOTO 71
70	      CONTINUE
71	      IF=1
	      DO 72 K=J+1,IBASE+LLENG(I)
		IF((XLE.AND.X(K).LE.CMXLE).OR.
     *		   (XEQ.AND.X(K).EQ.CMXEQ).OR.
     *		   (XGE.AND.X(K).GE.CMXGE).OR.
     *		 (YLE.AND.X(IY+K).LE.CMYLE).OR.
     *		 (YEQ.AND.X(IY+K).EQ.CMYEQ).OR.
     *		 (YGE.AND.X(IY+K).GE.CMYGE))THEN
	          IF=0
	        ELSE
	          CALL TRANS(I,X(K),X(IY+K),XT,YT)
	          CALL PLOT(XT,YT,IF,0)
		  if (interrupted) goto 51 ! *jd* 8.3.92
	          CALL MARK(IM)
	          IF(IF.EQ.0)THEN
	            IF=1
	          END IF
	        END IF
72	      CONTINUE
	    END IF
	    IF(ITYPE(I).EQ.2.OR.ITYPE(I).EQ.5)CALL MARK(IMARK(I))
	  END IF
	if (interrupted) go to 51 !*jd* 8.3.92
50	IBASE=IBASE+LLENG(I)
51 	CALL COLOR(IPEN(0))
	RETURN

	ENTRY PLOTV(X,NXS,XF,YF)
	IPV=1
	GOTO 1000

	ENTRY VELVCT(X,NXS,XF,YF)
	IPV=2
	GOTO 1000
	END





