        SUBROUTINE VECFLD(U,V,NXS,NYS,PX,PY,XF,YF)
	real u,v,px,py,xf,yf
	integer nxs,nys
C**
C**    @(#)vecfld.F	1.5    9/21/89
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
*	*sh* 04-16-90 	Mod to retain VUSRLN for user defined vector lengths
*       *jd* 8.3.92     Mod to add interrupts
*	*jd* 12.92	Mod to handle fields of 0 length vectors
*	*sh* 12/8/93    corrected bug: vectors blow on divide by zero
C
C	ARRAY X HOLD 2 GRIDS, U(X) IN THE FIRST
C	NX BY NY GRID AND V(Y) IN THE SECOND
C
	DIMENSION U(NXS,NYS),V(NXS,NYS),PX(NXS),PY(NYS)
C
C
#ifdef unix
        include 'parampl5_dat.decl'
        include 'PARAMPL5.DAT'
        include 'lines_inc.decl'
        include 'LINES.INC'
        include 'pen_inc.decl'
        include 'PEN.INC'
        include 'cmrd_inc.decl'
        include 'CMRD.INC'
        include 'miss_inc.decl'
        include 'MISS.INC'
        include 'switch_inc.decl'
        include 'SWITCH.INC'
        include 'hd_inc.decl'
        include 'HD.INC'
        include 'axis_inc.decl'
        include 'AXIS.INC'
        include 'labcom_inc.decl'
        include 'LABCOM.INC'
        include 'vector_inc.decl'
        include 'VECTOR.INC'
        include 'xcurvilinear.cmn'
#else
        INCLUDE 'tmap_pplv11inc:tmap_PARAMPL5.DAT'
        INCLUDE 'tmap_pplv11inc:tmap_LINES.INC'
        INCLUDE 'tmap_pplv11inc:tmap_PEN.INC'
        INCLUDE 'tmap_pplv11inc:tmap_CMRD.INC'
        INCLUDE 'tmap_pplv11inc:tmap_MISS.INC'
        INCLUDE 'tmap_pplv11inc:tmap_SWITCH.INC'
        INCLUDE 'tmap_pplv11inc:tmap_HD.INC'
        INCLUDE 'tmap_pplv11inc:tmap_AXIS.INC'
        INCLUDE 'tmap_pplv11inc:tmap_LABCOM.INC'
        INCLUDE 'tmap_pplv11inc:tmap_VECTOR.INC'
#endif

	real xfuser,yfuser,xlousr,ylousr,zave,xt,yt
	real xx2,yy2,zz,vfact,xtm,ytm,xtx,ytx,xx1,yy1,vx,vy,vclen
	real thfact,xn,yn,xl,yl,dy,dx
	integer ic,i,j,lnblk,iln

	CHARACTER VECLAB*20
C
C	VECTOR,X,Y,VFACT,LABEL
C

C     Curvilinear coordinates declarations
        integer icurv, status, ii
        real x_curv(12), y_curv(12), curv_len
        real tail_i, tail_j, head_i, head_j
        real arrow_i, arrow_j

#ifdef unix
        include         'xinterrupt.cmn' ! 1/89 *sh*
        include         'ppl_in_ferret.cmn'
#else
        INCLUDE         'TMAP_PPLV11INC:XINTERRUPT.CMN' ! 1/89 *sh*
        INCLUDE         'TMAP_PPLV11INC:PPL_IN_FERRET.CMN'
#endif

*******************************************************************************

	    IF(VUSER)THEN
	    	XFUSER=1.
		YFUSER=1.
		YLOUSR=0.
		XLOUSR=0.
	    ELSE
	    	XFUSER=XF
		YFUSER=YF
		XLOUSR=XLO
		YLOUSR=YLO
	    ENDIF
	XT=VXLOC/XFUSER+XLOUSR
	YT=VYLOC/YFUSER+YLOUSR
	IF(VAUTO)THEN
c
c	use average vector length for scale if auto
c
	    IC=0
	    ZAVE=0.0
	    DO 10 I=1,NX,VSKPX
	    DO 10 J=1,NY,VSKPY
	    XX2=U(I,J)
	    YY2=V(I,J)
C
C	TEST FOR MISSING DATA
C
	    IF((ZLE.AND.XX2.LE.CMZLE).OR.
     *	       (ZEQ.AND.XX2.EQ.CMZEQ).OR.
     *	       (ZGE.AND.XX2.GE.CMZGE).OR.
     *	       (ZLE.AND.YY2.LE.CMZLE).OR.
     *	       (ZEQ.AND.YY2.EQ.CMZEQ).OR.
     *	       (ZGE.AND.YY2.GE.CMZGE))GOTO 10
	    ZZ=SQRT(XX2**2+YY2**2)
	    ZAVE=ZAVE+ZZ
	    IC=IC+1
10	    CONTINUE
	    IF(IC.EQ.0)IC=1
*	    Mod to handle field of length 0 vectors *jd* 12.92
	    if (zave .eq. 0.0) then 
	       zave = 0.5
	    else
	       ZAVE=ZAVE/FLOAT(IC)
	    endif
	    VFACT=VLEN/(2.0*ZAVE)
	    vusrln = (2.0*ZAVE)		!  *sh* 4/90
	ELSE
	    VFACT=VLEN/VUSRLN
	ENDIF
	CALL COLOR(IPEN(1))
	call trans(0,xmin,ymin,xtm,ytm)
	call trans(0,xmax,ymax,xtx,ytx)
!*sh*	DX=(xtx-xtm)/(NX-1)
!*sh*	DY=(ytx-ytm)/(NY-1)
	DX=(xtx-xtm)/MAX(NX-1,1)
	DY=(ytx-ytm)/MAX(NY-1,1)
	DO 100 I=1,NX,VSKPX
	DO 100 J=1,NY,VSKPY
	XX2=U(I,J)
	YY2=V(I,J)
C
C	TEST FOR MISSING DATA
C
	IF((ZLE.AND.XX2.LE.CMZLE).OR.
     *	   (ZEQ.AND.XX2.EQ.CMZEQ).OR.
     *	   (ZGE.AND.XX2.GE.CMZGE).OR.
     *	   (ZLE.AND.YY2.LE.CMZLE).OR.
     *	   (ZEQ.AND.YY2.EQ.CMZEQ).OR.
     *	   (ZGE.AND.YY2.GE.CMZGE))GOTO 100
C
C	TAIL LOCATION USER UNITS
C
	IF(JTYPE.EQ.-1)THEN
	    XX1 = xtm + DX*(I-1)
	    YY1 = ytm + DY*(J-1)
	ELSE
	    call trans(0,px(i),py(j),xx1,yy1)
	ENDIF

 50    IF (curvilinear) THEN
           tail_i = xx1
           tail_j = yy1
           x_curv(1)=xx1
           y_curv(1)=yy1

           CALL CURV_COORD(x_curv, y_curv, 1, 1.0, 1.0, status) ! xform the pts

           xx1=x_curv(1)
           yy1=y_curv(1)

        ENDIF

        IF (curvilinear) THEN

           vclen = SQRT(U(I,J)**2 + V(I,J)**2)

           IF (vclen .GT. 1.0) THEN

              curv_len = 0.0
              DO 60, ii=1, 500

                 head_i = tail_i + U(I,J)/10
                 head_j = tail_j + V(I,J)/10

                 x_curv(1)=head_i
                 y_curv(1)=head_j

                 CALL CURV_COORD(x_curv, y_curv, 1, 1.0, 1.0, status)

                 xx2=x_curv(1)
                 yy2=y_curv(1)
                 
                 CALL PLOT(XX1,YY1,0,0)    
                 CALL PLOT(XX2,YY2,1,0)

                 curv_len = curv_len + SQRT((xx2-xx1)**2 + (yy2-yy1)**2)

                 tail_i = head_i
                 tail_j = head_j
                 xx1 = xx2
                 yy1 = yy2
                 
                 IF (curv_len*2 .GE. vclen*VFACT) GOTO 70

 60           CONTINUE

C     Now for the arrowhead


C     
C     CALC REAL LENGTH
C     
 70           VX=U(I,J)
              VY=V(I,J)
C     
C     SET UP THE ARROW FACTOR HEAD LENGTH SCALE
C     
              IF(VCLEN.GT.ARMAX)THEN
                 THFACT=ARMAX*HFACT/VCLEN
              ELSE IF(VCLEN.LT.ARMIN)THEN
                 THFACT=ARMIN*HFACT/VCLEN
              ELSE
                 THFACT=HFACT
              ENDIF
C     
C     DRAW ARROW HEADS
C     
              arrow_i = head_i + (-VX-.35*VY)*THFACT/XF * 20
              arrow_j = head_j + (-VY+.35*VX)*THFACT/YF * 20
              CALL CURV_COORD(arrow_i, arrow_j, 1, 1.0, 1.0, status)
              CALL PLOT(arrow_i, arrow_j,1,0)
              CALL PLOT(XX2,YY2,1,0)
C     
              arrow_i = head_i + (-VX+.35*VY)*THFACT/XF * 20
              arrow_j = head_j + (-VY-.35*VX)*THFACT/YF * 20
              CALL CURV_COORD(arrow_i, arrow_j, 1, 1.0, 1.0, status)
              CALL PLOT(arrow_i, arrow_j,1,0)
              CALL PLOT(XX2,YY2,1,0)

           ENDIF

           GOTO 100

        ENDIF

C
C	HEAD LOCATION USER UNITS
C

        IF (curvilinear) THEN

           head_i = tail_i + xx2
           head_j = tail_j + yy2

           x_curv(1)=head_i
           y_curv(1)=head_j

           CALL CURV_COORD(x_curv, y_curv, 1, 1.0, 1.0, status) ! xform the pts

           xx2=x_curv(1)
           yy2=y_curv(1)

        ELSE

           XX2 = XX1 + XX2*VFACT/XF
           YY2 = YY1 + YY2*VFACT/YF

        ENDIF

C
C	CALC REAL LENGTH
C
	VX=(XX2-XX1)*XF
	VY=(YY2-YY1)*YF
	VCLEN=SQRT(VX**2+VY**2)
C
C	IF VECTOR IS TOO SMALL PLOT A POINT
C
	IF(VCLEN.LT.VMINLN)THEN
	    CALL PLOT(XX1,YY1,0,0)
	    CALL PLOT(XX1,YY1,1,0)
	    GOTO 100
	ENDIF
C
C	SET UP THE ARROW FACTOR HEAD LENGTH SCALE
C
	IF(VCLEN.GT.ARMAX)THEN
	    THFACT=ARMAX*HFACT/VCLEN
	ELSE IF(VCLEN.LT.ARMIN)THEN
	    THFACT=ARMIN*HFACT/VCLEN
	ELSE
	    THFACT=HFACT
	ENDIF
C
C	DRAW ARROW HEADS
C
	XN=XX2+(-VX-.35*VY)*THFACT/XF
	YN=YY2+(-VY+.35*VX)*THFACT/YF
	CALL PLOT(XN,YN,0,0)
	CALL PLOT(XX2,YY2,1,0)
C
	XN=XX2+(-VX+.35*VY)*THFACT/XF
	YN=YY2+(-VY-.35*VX)*THFACT/YF
	CALL PLOT(XN,YN,1,0)
	CALL PLOT(XX2,YY2,1,0)

C
	CALL PLOT(XX1,YY1,1,0)
100	if (interrupted) goto 3333 ! *jd* 8.3.92
	CONTINUE
C
C	DRAW SCALE  MAKE SCALE 1 INCH LONG AT XT,YT
C
	IF(.NOT.VKEY)THEN
	    CALL COLOR(IPEN(0))
	    RETURN
	ENDIF
	CALL WINDOW(0.,0.,0.,0.)
	XX1=XT
	YY1=YT
	XX2=XT+VLEN/XF
	YY2=YT
C
C	DRAW ARROW HEADS
C
	IF(VLEN.GT.ARMAX)THEN
	    THFACT=ARMAX*HFACT/VLEN
	ELSE IF(VLEN.LT.ARMIN)THEN
	    THFACT=ARMIN*HFACT/VLEN
	ELSE
	    THFACT=HFACT
	ENDIF
	XN=XX2-THFACT*VLEN/XF
	YN=YY2+.35*THFACT*VLEN/YF

	CALL PLOT(XN,YN,0,0)
	CALL PLOT(XX2,YY2,1,0)
C
	XN=XX2-THFACT*VLEN/XF
	YN=YY2-.35*THFACT*VLEN/YF
	CALL PLOT(XN,YN,1,0)
	CALL PLOT(XX2,YY2,1,0)
C
	CALL PLOT(XX1,YY1,1,0)
C
C	DRAW LABEL
C
	CALL COLOR(IPEN(0))
	WRITE (VECLAB,VFRMT) VLEN/VFACT
	ILN=LNBLK(VECLAB,20)
	XL = XX2 + .125/XF
	YL = YY2 - HLABS*0.5/YF
	CALL SYMBEL(XL,YL,0.,HLABS,ILN,VECLAB)
3333	IF(IWIND.GT.0)CALL WINDOW(XLO,YLO,XHI,YHI)
	RETURN
	END
