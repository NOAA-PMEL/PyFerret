	SUBROUTINE SETAX(X,NXS,XF,YF,XL,YL,XH,YH,NMX,NMY,
     *	XTI,XTL)
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
*
*	Modified J Davison 11.28.88 for compatibility to v1.1 PPL
*
*       Declarations added 9.6.88 J Davison                           
*	Mod for 4 year digits *jd* 11.21.95
*  
* v5.41 *acm* - 3/02  If autoscaling is done on an axis, after the
*                     scale settings are made, set XAXIS_MIN, XAXIS_MAX
*                     or YAXIS_MIN, YAXIS_MAX
*               Also remove VMS include statements.


        real            x,diff,rxax,rxmon,rxchr,thi,xti,xtl,xf,xl,xh  
        real            xmn,xmx,xt,yf,yl,yh,yt                        
                                                                      
        integer         nxs,iy,nmx,i,ibasex,ibasey,len,j,nmy          
                                                                      
        external        diff                                          
*       End declarations 9.6.88
*	
*	Declarations added 11.28.88
	real		xtx,ytx,xtm,ytm
*	End declarations 11.28.88

	external RANGE

C**
C**    @(#)setax.F	1.2    9/8/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
	DIMENSION X(NXS)
C	KTYPE = 0 -XY  1 -Z  2-VECTOR

        include 'parampl5_dat.decl'
        include 'PARAMPL5.DAT'
        include 'lines_inc.decl'
        include 'LINES.INC'
        include 'pen_inc.decl'
        include 'PEN.INC'
        include 'pltl_inc.decl'
        include 'PLTL.INC'
        include 'axis_inc.decl'
        include 'AXIS.INC'
        include 'hd_inc.decl'
        include 'HD.INC'
        include 'taxis_inc.decl'
        include 'TAXIS.INC'
        include 'axisl_inc.decl'
        include 'AXISL.INC'
        include 'miss_inc.decl'
        include 'MISS.INC'
        include 'cmrdl_inc.decl'
        include 'CMRDL.INC'

	LOGICAL CONTR,VECTR
	CHARACTER IT1TMP*14,IT2TMP*14

C	SETUP AXIS FOR ALL PLOT TYPES.
	CONTR=MAKEP.EQ.'CONTOUR'.OR.MAKEP.EQ.'VIEW'.or.makep.eq.'SHADE'
     *	.OR.MAKEP.EQ.'VECTOR'
	VECTR=MAKEP.EQ.'PLOTV'.OR.MAKEP.EQ.'PLOTUV'
     *	.OR.MAKEP.EQ.'VELVCT'
	IY=NXS/2
	IF(ITFLG.NE.0)THEN
	    IF (IAUTOT .EQ. 1 .AND.
     1	        (IFORM(1:3).EQ.'EPI' .OR. IFORM(1:3).EQ.'BIB' .or.
     2		iform(1:3).eq.'LDX')) THEN
		CALL FIXTIM (ITMIN,ITMAX,IT1,IT2,ITYPET)
		TLO = 1. - DIFF(IT1,ITSTRT)/DT
	    ELSE
		CALL FIXTIM (IT1,IT2,IT1TMP,IT2TMP,ITYPET)
		IT1=IT1TMP
		IT2=IT2TMP
		TLO = 1. - DIFF(IT1,ITSTRT)/DT
	    ENDIF
C
C	    CHANGE FROM 3-CHAR MON TO 1-CHAR MON IF INSUFFICIENT SPACE
C
	    RXAX=DIFF(IT1,IT2)/1440.
	    IF(ITFLG.GT.0)THEN
		RXMON=30.*XLEN/RXAX
	    ELSE
		RXMON=30.*YLEN/RXAX
	    ENDIF
	    RXCHR=TCSIZE*6.
	    IF (RXCHR .GT. RXMON) then
		itxtpe=3
		labelt=itxtpe*itxlbp
	    endif
	    THI=TLO+DIFF(IT1,IT2)/DT
	ENDIF
	call trans(0,xmin,ymin,xtm,ytm)
	call trans(0,xmax,ymax,xtx,ytx)
	IF(ITFLG.EQ.1)THEN
	    XTI=XHI
	    XTL=XLO
	    XHI=THI
	    XLO=TLO
	    XF=XLEN/(XHI-XLO)
	    IF(CONTR)THEN
		XL=(xtm-XLO)*XF
		XH=XLEN+(xtx-XHI)*XF
	    ENDIF
	ELSE
	    IF(CONTR)THEN
C	CONTOUR PLOT
		IF(IAUTOX.EQ.1)THEN
		    IF(ITYPEX.GT.1)THEN
			CALL RANGEL(xtm,xtx,XLO,XHI)
			XLO=ALOG10(ABS(XLO))
			XHI=ALOG10(ABS(XHI))
		    ELSE
			CALL RANGE(xtm,xtx,NTICX,XLO,XHI,XTIC)
		    ENDIF
		    IF(xtm.gt.xtx)THEN
			XF=XLO
			XLO=XHI
			XHI=XF
			XTIC=-XTIC
		    ENDIF
		ENDIF
		XF=XLEN/(XHI-XLO)
		XL=(xtm-XLO)*XF
		XH=XLEN+(xtx-XHI)*XF
		NMX=NMTCX
                   CALL GETSYM('XAXIS_MAX',xaxmaxval,NC,IER)

	    ELSE
		IF(IAUTOX.EQ.0)GOTO 100
		IF(VECTR)THEN
		    XMN=1.
		    XMX=LLENG(1)
		    DO 50 I=1,LNUM
			IF(XMX.LT.LLENG(I))XMX=LLENG(I)
50		    CONTINUE
		ELSE
		    XMN=1.E38
		    XMX=-XMN
		    IBASEX=0
		    IBASEY=IY
		    DO 10 I=1,LNUM
			LEN=LLENG(I)
			DO 20 J=1,LEN
			    IF((XLE.AND.X(IBASEX+J).LE.CMXLE).OR.
     *			    (XEQ.AND.X(IBASEX+J).EQ.CMXEQ).OR.
     *			    (XGE.AND.X(IBASEX+J).GE.CMXGE).OR.
     *			    (YLE.AND.X(IBASEY+J).LE.CMYLE).OR.
     *			    (YEQ.AND.X(IBASEY+J).EQ.CMYEQ).OR.
     *			    (YGE.AND.X(IBASEY+J).GE.CMYGE))GOTO 20
			    XT=X(IBASEX+J)*XFCT(I)+XOFF(I)
			    IF(XT.LT.XMN)XMN=XT
			    IF(XT.GT.XMX)XMX=XT
20			CONTINUE
			IBASEX=IBASEX+LEN
			IBASEY=IBASEY+LEN
10		    CONTINUE
		ENDIF
		IF(ITYPEX.GT.1)THEN
		    CALL RANGEL(XMN,XMX,XLO,XHI)
		    XLO=ALOG10(ABS(XLO))
		    XHI=ALOG10(ABS(XHI))
		ELSE
		    CALL RANGE(XMN,XMX,NTICX,XLO,XHI,XTIC)
		ENDIF
	    ENDIF
100	    XF=XLEN/(XHI-XLO)
	    NMX=NMTCX
	ENDIF
C	YAXIS
	IF(ITFLG.EQ.-1)THEN
	    XTI=YHI
	    XTL=YLO
	    YHI=THI
	    YLO=TLO
	    YF=YLEN/(YHI-YLO)
	    IF(CONTR)THEN
		YL=(ytm-YLO)*YF
		YH=YLEN+(ytx-YHI)*YF
	    ENDIF
	ELSE
	    IF(CONTR)THEN
C CONTOUR DATA
		IF(IAUTOY.EQ.1)THEN
		    IF(ITYPEY.GT.1)THEN
			CALL RANGEL(ytm,ytx,YHI,YLO)
			YHI=ALOG10(ABS(YHI))
			YLO=ALOG10(ABS(YLO))
		    ELSE
			CALL RANGE(ytm,ytx,NTICY,YLO,YHI,YTIC)
		    ENDIF
		    IF(ytm.gt.ytx)THEN
			YF=YLO
			YLO=YHI
			YHI=YF
			YTIC=-YTIC
		    ENDIF

                   CALL GETSYM('YAXIS_MAX',yaxval,NC,IER)
		ENDIF
		YF=YLEN/(YHI-YLO)
		YL=(ytm-YLO)*YF
		YH=YLEN+(ytx-YHI)*YF
		NMY=NMTCY
	    ELSE
		IF(IAUTOY.EQ.0)GOTO 200
		XMN=1.E38
		XMX=-XMN
		IBASEY=IY
		IBASEX=0
		DO 30 I=1,LNUM
		    DO 40 J=1,LLENG(I)
			IF((XLE.AND.X(IBASEX+J).LE.CMXLE).OR.
     *			(XEQ.AND.X(IBASEX+J).EQ.CMXEQ).OR.
     *			(XGE.AND.X(IBASEX+J).GE.CMXGE).OR.
     *			(YLE.AND.X(IBASEY+J).LE.CMYLE).OR.
     *			(YEQ.AND.X(IBASEY+J).EQ.CMYEQ).OR.
     *			(YGE.AND.X(IBASEY+J).GE.CMYGE))GOTO 40
			YT=X(IBASEY+J)*YFCT(I)+YOFF(I)
			IF(YT.LT.XMN)XMN=YT
			IF(YT.GT.XMX)XMX=YT
40		    CONTINUE
		    IBASEX=IBASEX+LLENG(I)
30		IBASEY=IBASEY+LLENG(I)
		IF(VECTR)THEN
		    IF(ABS(XMX).GT.ABS(XMN))THEN
			XMN=-ABS(XMX)
		    ELSE
			XMX=ABS(XMN)
		    ENDIF
		ENDIF
		IF(ITYPEY.GT.1)THEN
		    CALL RANGEL(XMN,XMX,YLO,YHI)
		    YLO=ALOG10(ABS(YLO))
		    YHI=ALOG10(ABS(YHI))
		ELSE
		    CALL RANGE(XMN,XMX,NTICY,YLO,YHI,YTIC)
		ENDIF
200		YF=YLEN/(YHI-YLO)
		NMY=NMTCY
	    ENDIF
	ENDIF

        IF (IAUTOX .NE. 0) CALL setax_syms('X', xlo, xhi)
        IF (IAUTOY .NE. 0) CALL setax_syms('Y', ylo, yhi)


	RETURN
	END
