	SUBROUTINE SETAX(X,NXS,XF,YF,XL,YL,XH,YH,NMX,NMY,
     *	XTI,XTL,its_ribbon)
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
* v5.41 *acm*   6/20/02 remove unneeded calls to GETSYM.
* V554  9/03 *acm* Fix precision bug double precision, for fine grids in 
*                  curvlinear coordinates. See also pltit.F, plotz.F
* V694  *acm*  1/15 Changes to allow two time axes on plot (F-T)
* V694  *acm*  5/15 Ticket 2283. Handling axis scaling for ribbon plots. Do auto-scale 
*                   using just the first line. The second is the color-by variable
* V695  *acm*  9/15 Changes for hour:min labeling on time axes
* V697  *acm*  1/16 Ticket 2344: if there was a time-plot underlay, will use the underlay's 
*                   time scaling for time overlay plots.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax.

	REAL*8		DIFF, dtd
        real            x,rxax,rxmon,rxchr,thi,xti,xtl
        real            xmn,xmx,xt,yt
        real*8		dxl, dxh, dxf, dyl, dyh, dyf
	logical		its_ribbon
                                                                      
        integer         nxs,iy,nmx,i,ibasex,ibasey,len,j,nmy    
	
	integer		lnum_lines
                                                                      
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

	dtd = dt
	dt_save = dt

C	SETUP AXIS FOR ALL PLOT TYPES.
c	CONTR=MAKEP.EQ.'CONTOUR'.OR.MAKEP.EQ.'VIEW'.or.makep.eq.'SHADE'
c     *	.OR.MAKEP.EQ.'VECTOR'

	CONTR=MAKEP.EQ.'CONTOUR'.OR.MAKEP.EQ.'VIEW'
     *		.OR.MAKEP.EQ.'VECTOR'.or.makep.eq.'SHADE'   
     *          .or.makep.eq.'FLOWLINE'
     *          .or.makep.eq.'FILL'.or.makep.eq.'PIXEL'

	VECTR=MAKEP.EQ.'PLOTV'.OR.MAKEP.EQ.'PLOTUV'
     *	.OR.MAKEP.EQ.'VELVCT'
	IY=NXS/2
	IF(ITFLG.NE.0)THEN
	    IF (IAUTOT .EQ. 1 .AND.
     1	        (IFORM(1:3).EQ.'EPI' .OR. IFORM(1:3).EQ.'BIB' .or.
     2		iform(1:3).eq.'LDX')) THEN
		IF (is_hours_axis .GT. 0) THEN ! Round off to the nearest minute
		   CALL FIXTIM (ITMIN,ITMAX,IT1,IT2,4)
		ELSE
		   CALL FIXTIM (ITMIN,ITMAX,IT1,IT2,ITYPET) ! to the hour
		ENDIF
	    ELSE
		IF (is_hours_axis .GT. 0) THEN ! Round off to the nearest minute
		   CALL FIXTIM (IT1,IT2,IT1TMP,IT2TMP,4)
		ELSE
		   CALL FIXTIM (IT1,IT2,IT1TMP,IT2TMP,ITYPET)
		ENDIF
		IT1=IT1TMP
		IT2=IT2TMP
	    ENDIF
	    TLO = 1. - DIFF(IT1,ITSTRT)/dtd
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
	    THI=TLO+DIFF(IT1,IT2)/dtd
	ENDIF


	IF(ITFLG.EQ.2)THEN
	    IF (IAUTOT .EQ. 1 .AND.
     1	        (IFORM(1:3).EQ.'EPI' .OR. IFORM(1:3).EQ.'BIB' .or.
     2		iform(1:3).eq.'LDX')) THEN
		CALL FIXTIM (ITMIN,ITMAX,IT1b,IT2b,ITYPETb) 
		TLOb = 1. - DIFF(IT1,ITSTRT)/dtd
	    ELSE
		CALL FIXTIM (IT1b,IT2b,IT1TMP,IT2TMP,ITYPETb)
		IT1b=IT1TMP
		IT2b=IT2TMP
		TLOb = 1. - DIFF(IT1b,ITSTRTb)/dtd
	    ENDIF
C
C	    CHANGE FROM 3-CHAR MON TO 1-CHAR MON IF INSUFFICIENT SPACE
C
	    RXAX=DIFF(IT1b,IT2b)/1440.
	    RXMON=30.*XLEN/RXAX
	    RXCHR=TCSIZE*6.
	    IF (RXCHR .GT. RXMON) then
		itxtpeb=3
		labeltb=itxtpeb*itxlbpb
	    endif
	    THIb=TLOb+DIFF(IT1b,IT2b)/dtd
	ENDIF

	call trans(0,xmin,ymin,xtm,ytm)
	call trans(0,xmax,ymax,xtx,ytx)
	
	IF(ITFLG.EQ.2)THEN
	    XTI=XHI
	    XTL=XLO
	    XHI=THI
	    XLO=TLO
	    DXF=XLEN/(XHI-XLO)
	    IF(CONTR)THEN
		DXL=(xtm-XLO)*DXF
		DXH=XLEN+(xtx-XHI)*DXF
	    ENDIF
	ENDIF
	IF(ITFLG.EQ.1)THEN
	    XTI=XHI
	    XTL=XLO
	    XHI=THI
	    XLO=TLO
	    DXF=XLEN/(XHI-XLO)
	    IF(CONTR)THEN
		DXL=(xtm-XLO)*DXF
		DXH=XLEN+(xtx-XHI)*DXF
	    ENDIF
	ELSE
	    IF(CONTR)THEN
C	CONTOUR PLOT
		IF(IAUTOX.EQ.1)THEN
		    IF(ITYPEX.GT.1)THEN
			CALL RANGEL(xtm,xtx,XLO,XHI)
			XLO=LOG10(ABS(XLO))
			XHI=LOG10(ABS(XHI))
		    ELSE
			CALL RANGE(xtm,xtx,NTICX,XLO,XHI,XTIC)
		    ENDIF
		    IF(xtm.gt.xtx)THEN
			DXF=XLO
			XLO=XHI
			XHI=DXF
			XTIC=-XTIC
		    ENDIF
		ENDIF
		DXF=XLEN/(XHI-XLO)
		DXL=(xtm-XLO)*DXF
		DXL=(xtm-XLO)*DXF
		DXH=XLEN+(xtx-XHI)*DXF
		NMX=NMTCX              
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
		    lnum_lines = lnum
		    IF (its_ribbon) lnum_lines = 1  
		    DO 10 I=1,lnum_lines
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
		    XLO=LOG10(ABS(XLO))
		    XHI=LOG10(ABS(XHI))
		ELSE
		    CALL RANGE(XMN,XMX,NTICX,XLO,XHI,XTIC)
		ENDIF
	    ENDIF
100	    DXF=XLEN/(XHI-XLO)
	    NMX=NMTCX
	ENDIF
C	YAXIS
	IF(ITFLG.EQ.2)THEN
	    XTI=YHI
	    XTL=YLO
	    YHI=THIb
	    YLO=TLOb
	    DYF=YLEN/(YHI-YLO)
	    IF(CONTR)THEN
		DYL=(ytm-YLO)*DYF
		DYH=YLEN+(ytx-YHI)*DYF
	    ENDIF
        ENDIF

	IF(ITFLG.EQ.-1)THEN
	    XTI=YHI
	    XTL=YLO
	    YHI=THI
	    YLO=TLO
	    DYF=YLEN/(YHI-YLO)
	    IF(CONTR)THEN
		DYL=(ytm-YLO)*DYF
		DYH=YLEN+(ytx-YHI)*DYF
	    ENDIF
	ELSE
	    IF(CONTR)THEN
C CONTOUR DATA
		IF(IAUTOY.EQ.1)THEN
		    IF(ITYPEY.GT.1)THEN
			CALL RANGEL(ytm,ytx,YHI,YLO)
			YHI=LOG10(ABS(YHI))
			YLO=LOG10(ABS(YLO))
		    ELSE
			CALL RANGE(ytm,ytx,NTICY,YLO,YHI,YTIC)
		    ENDIF
		    IF(ytm.gt.ytx)THEN
			DYF=YLO
			YLO=YHI
			YHI=DYF
			YTIC=-YTIC
		    ENDIF
		ENDIF
		DYF=YLEN/(YHI-YLO)
		DYL=(ytm-YLO)*DYF
		DYH=YLEN+(ytx-YHI)*DYF
		NMY=NMTCY
	    ELSE
		IF(IAUTOY.EQ.0)GOTO 200
		XMN=1.E38
		XMX=-XMN
		IBASEY=IY
		IBASEX=0
		lnum_lines = lnum
		IF (its_ribbon) lnum_lines = 1  
		DO 30 I=1,lnum_lines
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
		    IBASEY=IBASEY+LLENG(I)
30		CONTINUE
		IF(VECTR)THEN
		    IF(ABS(XMX).GT.ABS(XMN))THEN
			XMN=-ABS(XMX)
		    ELSE
			XMX=ABS(XMN)
		    ENDIF
		ENDIF
		IF(ITYPEY.GT.1)THEN
		    CALL RANGEL(XMN,XMX,YLO,YHI)
		    YLO=LOG10(ABS(YLO))
		    YHI=LOG10(ABS(YHI))
		ELSE
		    CALL RANGE(XMN,XMX,NTICY,YLO,YHI,YTIC)
		ENDIF
200		DYF=YLEN/(YHI-YLO)
		NMY=NMTCY
	    ENDIF
	ENDIF

C	YAXIS and XAXIS are time/forecast axes
	IF(ITFLG.EQ.2)THEN
	    XTI=YHI
	    XTL=YLO
	    YHI=THIb
	    YLO=TLOb
	    DYF=YLEN/(YHI-YLO)
	    IF(CONTR)THEN
		DYL=(ytm-YLO)*DYF
		DYH=YLEN+(ytx-YHI)*DYF
	    ENDIF
	ENDIF
 
        IF (IAUTOX .NE. 0) CALL setax_syms('X', xlo, xhi)
        IF (IAUTOY .NE. 0) CALL setax_syms('Y', ylo, yhi)

	XF = DXF
	YF = DYF
	XL = DXL
	YL = DYL
	XH = DXH
	YH = DYH
	
	RETURN
	END
