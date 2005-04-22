        SUBROUTINE VECFLD(U,V,NXS,NYS,PX,PY,XF,YF)
        real u,v,px,py,xf,yf
        integer nxs,nys
C**
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
*       *acm*7/01	apply scaling to vectors when curv. coords.
*       *acm*		NOTE: If this routine plots vectors directly along
*                       the "edge" of the region for modulo axes, try setting
*                       the region to one grid point less than the whole
*                       globe, e.g. /X=0:357.5   When curv_coord_sub.F applies
*			the modulo correction it "interpolates" between two 
*			x values that are the same.  Also see notes in 
*                       ~ansley/ans_ferret/curvilinear/curv_coord_sub.F
*       *acm* 10/01    Remove VMS includes
*       *acm* 10/24/01 Call mark(99) to plot a point when a vector is too small.
*       *acm*  5/02    Bug fix: alen for curvilinear vectors could be 0.
*       *ACM*  8/02    remove the call to plot a point with mark(99) -- need
*                      to resolve PPLUS issues with polymark first
*       *acm*  9/02    Yet another tweak on the length of curvlinear vectors.
*                      Fix the computation of curv_len, page length of vector as 
*                      it is drawn.
*       *acm* 12/02    Restore call to mark(99) polymarker call when vector is
*                      small.  Call doit_makedot after the loop, to plot points in the buffer.
* v5.81 *acm* lengthen string containing format to 40 characters.

C	ARRAY X HOLD 2 GRIDS, U(X) IN THE FIRST
C	NX BY NY GRID AND V(Y) IN THE SECOND
C
	DIMENSION U(NXS,NYS),V(NXS,NYS),PX(NXS),PY(NYS)
C
C
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
        include 'makedot.cmn'

	real xfuser,yfuser,xlousr,ylousr,zave,xt,yt
	real xx2,yy2,zz,vfact,xtm,ytm,xtx,ytx,xx1,yy1,vx,vy,vclen
	real thfact,xn,yn,xl,yl,dy,dx, alen

	integer ic,i,j,lnblk,iln, overedge

	CHARACTER VECLAB*40
C
C	VECTOR,X,Y,VFACT,LABEL
C

C     Curvilinear coordinates declarations
        integer status, ii
        real x_curv(12), y_curv(12), curv_len, xxc, yyc
        real tail_i, tail_j, head_i, head_j, head_len

        include         'xinterrupt.cmn' ! 1/89 *sh*
        include         'ppl_in_ferret.cmn'

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
	    IF (zave .eq. 0.0) THEN 
	       zave = 0.5
	    else
	       ZAVE=ZAVE/FLOAT(IC)
	    ENDIF
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
        IPT = 0
	DO 100 I=1,NX,VSKPX
	DO 100 J=1,NY,VSKPY

        IF (IPT .GT. maxdot) THEN
           CALL doit_makedot(IPT,XDOT,YDOT)
           IPT = 0
        ENDIF

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

C  vclen is approx vector length; without curviness.

           vclen = SQRT( (U(I,J)**2) + (V(I,J)**2))

C Set up to plot the vector in pieces to make a curve.

           tail_i = xx1
           tail_j = yy1
           x_curv(1)=xx1
           y_curv(1)=yy1
           CALL CURV_COORD(x_curv, y_curv, 1, 1.0, 1.0, status) ! xform the pts

           xx1=x_curv(1)
           yy1=y_curv(1)
           xxc=x_curv(1)	! initialize, in case of bad data in loop 60 below
           yyc=y_curv(1)
           xx1orig = xx1
           yy1orig = yy1

           curv_len = 0.0
           overedge = 0

           DO 60, ii=1, 500

              head_i = tail_i + U(I,J)*vfact/10
              head_j = tail_j + V(I,J)*vfact/10

              IF (head_i .GT. xtx .AND. (.NOT. dim1_modulo)) THEN
                 head_i = xtx
                 overedge = 1
              ENDIF
              IF (head_j .GT. ytx .AND. (.NOT. dim2_modulo)) THEN
                 head_j = ytx
                 overedge = 1
              ENDIF
              IF (head_i .LT. xtm .AND. (.NOT. dim1_modulo)) THEN
                 head_i = xtm
                 overedge = 1
              ENDIF
              IF (head_j .LT. ytm .AND. (.NOT. dim2_modulo)) THEN
                 head_j = ytm
                 overedge = 1
              ENDIF

              x_curv(1) = head_i
              y_curv(1) = head_j

              CALL CURV_COORD(x_curv, y_curv, 1, 1.0, 1.0, status)

              IF (x_curv(1) .ne. cmzeq) xxc = x_curv(1)
              IF (y_curv(1) .ne. cmzeq) yyc = y_curv(1)

              CALL PLOT(XX1,YY1,0,0)    
              CALL PLOT(xxc,yyc,1,0)

C  curv_len is sum of vector pieces; actual vector length.  When 
C  it is vclen we are done plotting the vector.

C   Note: len = distance from xx1 to xxc, which in page units is
C   equivalent to xxc = xx1 + (len/xf), or len = (xxc-xx1)* xf

              curv_len = curv_len + 
     .           SQRT( ( (xxc-xx1)*xf )**2  +  ( (yyc-yy1)*yf )**2 )

              IF (overedge .EQ. 1) GO TO 80 

              IF (curv_len .GE. vclen* vfact) GO TO 70

              tail_i = head_i
              tail_j = head_j
              xx1 = xxc
              yy1 = yyc

 60        CONTINUE

 70        CONTINUE

C     DRAW ARROW HEADS

	   IF (xxc .NE. cmzeq  .AND. yyc .NE. cmzeq) THEN
              head_len = 2.*curv_len/(xf+yf)
              CALL vec_arrows(tail_i, tail_j, head_i, head_j, head_len)
	   ENDIF	! draw arrow if xxc and yyc were ok
   80	   CONTINUE	! vector went over edge

        ELSE	! not curvilinear

           XX2 = XX1 + XX2*VFACT/XF
           YY2 = YY1 + YY2*VFACT/YF
C
C	CALC REAL LENGTH
C
           VX=(XX2-XX1)*XF
           VY=(YY2-YY1)*YF
           VCLEN=SQRT(VX**2+VY**2)
C
C	IF VECTOR IS TOO SMALL PLOT A POINT
C
           IF(VCLEN .LT. VMINLN)THEN
	      CALL PLOT(XX1,YY1,0,0)
	      CALL PLOT(XX1,YY1,1,0)
              CALL MARK(99)
	      GOTO 100
           ENDIF
C
C	SET UP THE ARROW FACTOR HEAD LENGTH SCALE
C
           IF(VCLEN .GT. ARMAX)THEN
	      THFACT=ARMAX*HFACT/VCLEN
           ELSE IF(VCLEN .LT. ARMIN .AND. VCLEN.GT.0.)THEN
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
        ENDIF

100	IF (interrupted) goto 3333 ! *jd* 8.3.92
	CONTINUE
        CALL doit_makedot(IPT,XDOT,YDOT)
        IPT = 0

C
C	DRAW SCALE  MAKE SCALE .5 INCH LONG AT XT,YT
C
	IF(.NOT.VKEY)THEN
	    CALL COLOR(IPEN(0))
	    RETURN
	ENDIF
	CALL WINDOW(0.,0.,0.,0.)

C Note don't need different arrow length for curvilinear case.
C it is a standard half inch.

	XX1=XT
	YY1=YT
	XX2=XT+VLEN/XF
	YY2=YT
        alen = vlen
C
C	DRAW ARROW 
C
	IF(alen.GT.ARMAX)THEN
	    THFACT=ARMAX*HFACT/alen
	ELSE IF(alen.LT.ARMIN .AND. alen .GT. 1.e-6)THEN
	    THFACT=ARMIN*HFACT/alen
	ELSE
	    THFACT=HFACT
	ENDIF
	XN=XX2-THFACT*alen/XF
	YN=YY2+.35*THFACT*alen/YF

	CALL PLOT(XN,YN,0,0)
	CALL PLOT(XX2,YY2,1,0)
C
	XN=XX2-THFACT*alen/XF
	YN=YY2-.35*THFACT*alen/YF
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
