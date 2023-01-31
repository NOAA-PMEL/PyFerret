      SUBROUTINE flowline_sub (u, v, startflag, siz, den, nxs, nys,
     .                         px, py, xf, yf)


C**
C**    @(#)flowline_sub.F	    10/03/00
C**
C**
C***********************************************************************
C**
C**              PLOT+ Scientific Graphics System
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
*  ACM  From vecfld.F  9/00
*       and from GrADS module gxstrm.c
*       Draw flowlines based on a velocity field.
*  ACM  June 08, 2001  remove extra declaration of i, and
*       change MAX(1, x + 1) to MAX(1., x + 1)
*  ACM  Aug 2001 remove subroutine strmar to separate file (arrow-drawing)
*      *acm* 10/01  remove VMS includes
*  ACM  12/01 bug fix: jjndex not jjindex!
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus
* V686 *acm* 11/13 Allow symbol names to be up to 120 characters long

        IMPLICIT NONE
        include 'ferret.parm'
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
        include         'xinterrupt.cmn' ! 1/89 *sh*
        include         'ppl_in_ferret.cmn'


*******************************************************************************



C	ARRAY X HOLD 2 GRIDS, U(X) IN THE FIRST
C	NX BY NY GRID AND V(Y) IN THE SECOND
C       startflag is a work array.
C
C  Calling argument declarations

      INTEGER siz, den, nxs, nys
      REAL xf, yf, px(nxs),py(nys), u(nx,ny), v(nx,ny), startflag(*)

C  From from GrADS module gxstrm.c
C See http://grads.sourcearchive.com/documentation/2.0.a7.1-3/main.html 
C void gxstrm (float *u, float *v, float *c, int is, int js,
C   float uund, float vund, float cund, int flag, float *shdlvs,
C   int *shdcls, int shdcnt, int den)

C  Changes:
C  Trans to Fortran
C  No colorizing the lines
C  Curvlinear coordinate transformation
C  Allow irregular grids  (ITYPE not -1)

C  Local declarations
      INTEGER is, js

      REAL up, vp, x, y, xx, yy, uv1, uv2, uv, vv1, vv2, vv, auv, avv, 
     .     xsav, ysav, xold, yold, fact, rscl, xxsv, yysv, xtm, ytm, 
     .     xtx, ytx, upsv, vpsv, xup, yup, xinterp, yinterp, 
     .     vel, velmean, sqsum, cos_factor, dx, dy, count, sd, xxc, yyc,
     .     upi, upj, upij, vpi, vpj, vpij, xfuser, yfuser, ylousr,
     .     xlousr, xt, yt, xx1, yy1, xl, yl, vfact
      INTEGER LNBLK, TM_LENSTR1, 
     .     i, i1, ii, jj, ii1, ij1, i2, j2, mx, my, ipt, acnt, icol, 
     .     scol, dis, iindex, jjndex, ii1index, iacc, iisav, iscl, imn, 
     .     jmn, j, iz, jz, iss, jss, imx, jmx, iln, slen
      LOGICAL TM_FPEQ_SNGL, bflg

      CHARACTER*120 veclab, buff, symbuff
      EXTERNAL lnblk

C     Curvilinear coordinates declarations
      INTEGER status
      REAL*8 x_curv, y_curv

C First check that the curvilinear coordinate data is all present and accounted for.

      curv_missing = .FALSE.
      IF (curvilinear) THEN
         do j = 1, ny
         do i = 1, nx
            x_curv = i
	    y_curv = j
	    CALL curv_coord (x_curv, y_curv, 1, 1.0, 1.0, status)
	    IF (curv_missing) GOTO 5100
	 enddo
	 enddo
      ENDIF

C  From vector
      CALL COLOR(IPEN(1))

*  Get offset for true grid

      CALL trans(0,xmin,ymin,xtm,ytm)
      CALL trans(0,xmax,ymax,xtx,ytx)

      dx=(xtx-xtm)/MAX(nx-1,1)
      dy=(ytx-ytm)/MAX(ny-1,1)

      scol = -9
      icol = 1

      is = nx
      js = ny

C   Figure out the interval for the flag grid

      i = nx
      IF (ny .GT. i) i = ny
      iscl = 200/i
      iscl = iscl + den - 5
      if (iscl .LT. 1) iscl=1
      if (iscl .GT. 10) iscl=10
      fact = 0.5/REAL(iscl)
      rscl = REAL(iscl)
 
      iss = nx*iscl
      jss = ny*iscl

C  Get mean velocity for scaling the arrows

      sqsum = 0.
      vel = 0.
      velmean = 0.
      count = 0.
      DO i = 1, nx
         DO j = 1, ny 
            up = u(i,j)
            vp = v(i,j)
            IF (ZLE .AND. (up .LE. CMZLE .OR. vp .LE. CMZLE)) GO TO 89
            IF (ZEQ .AND. (up .EQ. CMZEQ .OR. vp .EQ. CMZEQ)) GO TO 89
            IF (ZGE .AND. (up .GE. CMZGE .OR. vp .GE. CMZGE)) GO TO 89

            vel = up*up + vp*vp
            sqsum = sqsum + vel
            vel = sqrt(vel)

            vel = max(vel, vel)
            velmean = velmean + vel
            count = count + 1.

  89        CONTINUE
         ENDDO
      ENDDO 
      IF (count .GT. 0.) THEN
        velmean = velmean/ count
        sd = sqrt(sqsum/ count - velmean* velmean)
      ENDIF

      IF (vauto) THEN
         vfact = 2./velmean
      ELSE
         vfact = 2./vusrln
      ENDIF

C  Initialize the flag grid

      DO i = 1, siz
         startflag(i) = 0.
      ENDDO

C   Loop through flag grid to look for start of flowlines.  
C     To start requires no flowlines drawn within surrounding 
C     flag boxes. 

      i2 = 1
      j2 = 1
      DO i=1, siz
         dis = 2
         IF (den .LT. 5) dis = 3
         IF (den .GT. 5) dis = 1
         imn = i2 - dis
         imx = i2 + dis + 1 
         jmn = j2 - dis
         jmx = j2 + dis + 1
         IF (imn .LT. 1) imn = 1
         IF (imx .GT. iss) imx = iss
         IF (jmn .LT. 1) jmn = 1
         IF (jmx .GT. jss) jmx = jss
         iacc = 0
         DO jz = jmn, jmx
            ipt = jz*iss + imn
            DO iz = imn, imx
              iacc = iacc + INT(startflag(ipt))
              ipt = ipt + 1
            ENDDO
         ENDDO
         IF (iacc .EQ. 0) THEN
            x = REAL((i2-1))/rscl
            y = REAL((j2-1))/rscl

            xsav = x
            ysav = y

            IF(JTYPE.EQ.-1)THEN
               xx = dx* x + xtm
               yy = dy* y + ytm
            ELSE
               mx = INT(x)
               my = INT(y) 
               xup = x - FLOAT(mx)
               yup = y - FLOAT(my)
               IF (mx+2 .LE. nxs) THEN
                 xinterp = px(mx+1) + xup* (px(mx+2) - px(mx+1))
               ELSE
                  xinterp = px(nxs)
               ENDIF

               IF (my+2 .LE. nys) THEN
                  yinterp = py(my+1) + yup* (py(my+2) - py(my+1))
               ELSE
                  yinterp = py(nys)
               ENDIF

	       call trans(0,xinterp,yinterp,xx,yy)
            ENDIF

            IF (curvilinear) THEN
               x_curv = xx
               y_curv = yy
               CALL curv_coord (x_curv, y_curv, 1, 1.0, 1.0, status) ! xform the pts

               xxc=x_curv
               yyc=y_curv
               CALL plot (xxc,yyc,0,0)
            ELSE
               CALL plot (xx,yy,0,0)
            ENDIF

            xxsv = xx 
            yysv = yy
            iisav = -999
            iacc = 0
            acnt = 0
            bflg = .FALSE.

            DO WHILE (x .GT. 0.0 .AND. x .LT. REAL(is-1) .AND. 
     .                y .GT. 0.0 .AND. y .LT. REAL(js-1)) 
               ii = INT(x) 
               jj = INT(y)
               xx = x - REAL(ii)
               yy = y - REAL(jj)

               iindex = ii + 1		! ACM: indices must start at 1 not 0.
               jjndex = jj + 1

               IF (jtype .NE. -1) THEN 

                  IF (iindex .LT. nxs) THEN
                     dx = px(iindex+1) - px(iindex)
                  ELSE 
                     dx = px(iindex) - px(iindex-1)
                  ENDIF
                  IF (jjndex .LT. nys) THEN
                     dy = py(jjndex+1) - py(jjndex)
                  ELSE 
                     dy = py(jjndex) - py(jjndex-1)
                  ENDIF

               ENDIF

               up  = u(iindex,jjndex)
               upi = u(iindex+1,jjndex)
               upj = u(iindex,jjndex+1)
               upij= u(iindex+1,jjndex+1)

               vp  = v(iindex,jjndex)
               vpi = v(iindex+1,jjndex)
               vpj = v(iindex,jjndex+1)
               vpij= v(iindex+1,jjndex+1)

               IF (ZLE .AND.
     .               (up .LE. CMZLE .OR. upi .LE. CMZLE .OR.
     .                upj.LE. CMZLE .OR. upij.LE. CMZLE .OR.          
     .                vp .LE. CMZLE .OR. vpi .LE. CMZLE .OR.
     .                vpj.LE. CMZLE .OR. vpij.LE. CMZLE)) GO TO 19

               IF (ZEQ .AND.
     .               (up .EQ. CMZEQ .OR. upi .EQ. CMZEQ .OR.
     .                upj.EQ. CMZEQ .OR. upij.EQ. CMZEQ .OR.          
     .                vp .EQ. CMZEQ .OR. vpi .EQ. CMZEQ .OR.
     .                vpj.EQ. CMZEQ .OR. vpij.EQ. CMZEQ)) GO TO 19

               IF (ZGE .AND.
     .               (up .GE. CMZGE .OR. upi .GE. CMZGE .OR.
     .                upj.GE. CMZGE .OR. upij.GE. CMZGE .OR.          
     .                vp .GE. CMZGE .OR. vpi .GE. CMZGE .OR.
     .                vpj.GE. CMZGE .OR. vpij.GE. CMZGE)) GO TO 19

               uv1 = up  + (upi - up)* xx
               uv2 = upj + (upij - upj)* xx
               uv  = uv1 + (uv2 - uv1)* yy

               vv1 = vp  + (vpi - vp)* xx
               vv2 = vpj + (vpij - vpj)* xx
               vv  = vv1 + (vv2-vv1)*yy

               auv = ABS(uv) 
               avv = ABS(vv)
               IF (auv .LT. 0.001 .AND. avv .LT. 0.001) GO TO 19

               IF (auv .GT. avv) THEN
                  uv = uv*fact/auv
                  vv = vv*fact/auv
               ELSE
                  uv = uv*fact/avv
                  vv = vv*fact/avv
               ENDIF

C From re_orient.F
C will we need COS(latitude) corrections ? ( the true dx may be dx*COS(lat) )
C compute COS(lat) correction at this latitude

               IF ( need_cos ) THEN
  
                  IF (JTYPE.EQ.-1)THEN
                     xx = dx* x + xtm
                     yy = dy* y + ytm
                  ELSE
                     mx = INT(x)
                     my = INT(y) 
                    xinterp = px(mx+1)
                     yinterp = py(my+1)
                     call trans(0,xinterp,yinterp,xx,yy)
                  ENDIF

                  cos_factor = COS( deg2rad * yy )
                  IF (trnspz) cos_factor = 1./COS( deg2rad * xx )
                  IF ( cos_factor .LT. 1.E-6 ) cos_factor = 1.E-6 !arbitrary fix /0
               ELSE
                 cos_factor = 1.0
               ENDIF

               x = x + uv/cos_factor
               y = y + vv/cos_factor

C  Correction for irregular grids and rectangular regions
C  and for latitude.

               IF (jtype .NE. -1) THEN	! get local dx, dy
                  mx = MAX(1., x + 1)
                  my = MAX(1., y + 1)
                  IF (mx .LT. nxs) THEN
                     dx = px(mx+1) - px(mx)
                  ELSE 
                     dx = px(mx) - px(mx-1)
                  ENDIF
                  IF (my .LT. nys) THEN
                     dy = py(my+1) - py(my)
                  ELSE 
                     dy = py(my) - py(my-1)
                  ENDIF
               ENDIF

               IF (.NOT. curvilinear) THEN
                  IF (.NOT. trnspz) THEN
                     x = x + uv/ (dx*cos_factor/dy)
                     y = y + vv/ (dy/(dx*cos_factor))
                  ELSE
                     x = x + uv/ (dx/(dy*cos_factor))
                     y = y + vv/ (dy*cos_factor/dx)
                  ENDIF
               ENDIF

               ii1 = INT(x*rscl)
               ij1 = INT(y*rscl)
               ii1 = ij1*iss + ii1
               ii1index = ii1 + 1

               IF (ii1 .LT. 0  .OR.  ii1 .GE. siz) GO TO 19
               IF (startflag(ii1index) .EQ. 1.) GO TO 19

               IF (ii1index .NE. iisav .AND. iisav.GT.0)
     .             startflag(iisav) = 1.
               IF (ii1index .EQ. iisav) THEN
                  iacc = iacc + 1
               ELSE
                  iacc = 0
               ENDIF

               IF (iacc .GT. 10) GO TO 19

               iisav = ii1index

               IF (JTYPE.EQ.-1)THEN
                  xx = dx* x + xtm
                  yy = dy* y + ytm

               ELSE
                  mx = INT(x)
                  my = INT(y) 
                  xup = x - FLOAT(mx)
                  yup = y - FLOAT(my)
                  IF (mx+2 .LE. nxs) THEN
                     xinterp = px(mx+1) + xup* (px(mx+2) - px(mx+1))
                  ELSE
                     xinterp = px(nxs)
                  ENDIF

                  IF (my+2 .LE. nys) THEN
                     yinterp = py(my+1) + yup* (py(my+2) - py(my+1))
                  ELSE
                     yinterp = py(nys)
                  ENDIF

                  call trans(0,xinterp,yinterp,xx,yy)
               ENDIF

               IF (icol .GT. -1) THEN
                  IF (bflg) THEN
                     
                     IF (curvilinear) THEN
                        x_curv = xold
                        y_curv = xold
                        CALL curv_coord (x_curv, y_curv, 1, 1.0, 1.0, 
     .                      status) ! xform the pts
                        xxc=x_curv
                        yyc=y_curv
                        CALL plot (xxc,yyc,0,0)
                     ELSE
                        CALL plot(xold,yold,0,0)
                     ENDIF

                     bflg = .FALSE.
                  ENDIF

                  IF (curvilinear) THEN
                     x_curv = xx
                     y_curv = yy
                     CALL curv_coord (x_curv, y_curv, 1, 1.0, 1.0, 
     .                   status) ! xform the pts
                     xxc=x_curv
                     yyc=y_curv
                     CALL plot (xxc,yyc,1,0)
                  ELSE
                     CALL plot(xx,yy,1,0)

                  ENDIF
               ELSE
                  bflg = .TRUE.
               ENDIF

               xold = xx
               yold = yy
               acnt = acnt + 1

C  Draw arrow heads

               IF (acnt .GT. 20) THEN

                 vel = SQRT(upsv*upsv + vpsv*vpsv)* vfact
                 IF (icol .GT. -1) 
     .              CALL strmar (xxsv, yysv, xx, yy, vel, xf, yf)
                  acnt = 0
               ENDIF

               xxsv = xx 
               yysv = yy
               upsv = up
               vpsv = vp
            ENDDO

   19       CONTINUE
            bflg = .FALSE.

C  The curve was stopped (either vel too small, or crosses another curve
C  or runs into an edge or bad data)  Go the other direction from where
C  we began the curve.

            x = xsav 
            y = ysav
            xx = dx* x + xtm
            yy = dy* y + ytm

            IF(JTYPE.EQ.-1)THEN
               xx = dx* x + xtm
               yy = dy* y + ytm
            ELSE
               mx = INT(x)
               my = INT(y) 
               xup = x - FLOAT(mx)
               yup = y - FLOAT(my)
               IF (mx+2 .LE. nxs) THEN
                  xinterp = px(mx+1) + xup* (px(mx+2) - px(mx+1))
               ELSE
                  xinterp = px(nxs)
               ENDIF

               IF (my+2 .LE. nys) THEN
                  yinterp = py(my+1) + yup* (py(my+2) - py(my+1))
               ELSE
                  yinterp = py(nys)
               ENDIF

	       call trans(0,xinterp,yinterp,xx,yy)
            ENDIF

            IF (curvilinear) THEN
               x_curv = xx
               y_curv = yy
               CALL curv_coord (x_curv, y_curv, 1, 1.0, 1.0, status) ! xform the pts
               xxc=x_curv
               yyc=y_curv
               CALL plot (xxc,yyc,0,0)
            ELSE
               CALL plot (xx,yy,0,0)
            ENDIF

            xxsv = xx
            yysv = yy

            iisav = -999
            iacc = 0
            acnt = 19

            DO WHILE (x .GT. 0.0 .AND. x .LT. REAL(is-1) .AND. 
     .                y .GT. 0.0 .AND. y .LT. REAL(js-1)) 
               ii = INT(x)
               jj = INT(y)
               xx = x - REAL(ii)
               yy = y - REAL(jj)

               iindex = ii + 1
               jjndex = jj + 1

               up  = u(iindex,  jjndex)
               upi = u(iindex+1,jjndex)
               upj = u(iindex,  jjndex+1)
               upij= u(iindex+1,jjndex+1)

               vp  = v(iindex,  jjndex)
               vpi = v(iindex+1,jjndex)
               vpj = v(iindex,  jjndex+1)
               vpij= v(iindex+1,jjndex+1)

               IF (ZLE .AND.
     .               (up .LE. CMZLE .OR. upi .LE. CMZLE .OR.
     .                upj.LE. CMZLE .OR. upij.LE. CMZLE .OR.          
     .                vp .LE. CMZLE .OR. vpi .LE. CMZLE .OR.
     .                vpj.LE. CMZLE .OR. vpij.LE. CMZLE)) GO TO 29

               IF (ZEQ .AND.
     .               (up .EQ. CMZEQ .OR. upi .EQ. CMZEQ .OR.
     .                upj.EQ. CMZEQ .OR. upij.EQ. CMZEQ .OR.          
     .                vp .EQ. CMZEQ .OR. vpi .EQ. CMZEQ .OR.
     .                vpj.EQ. CMZEQ .OR. vpij.EQ. CMZEQ)) GO TO 29

               IF (ZGE .AND.
     .               (up .GE. CMZGE .OR. upi .GE. CMZGE .OR.
     .                upj.GE. CMZGE .OR. upij.GE. CMZGE .OR.          
     .                vp .GE. CMZGE .OR. vpi .GE. CMZGE .OR.
     .                vpj.GE. CMZGE .OR. vpij.GE. CMZGE)) GO TO 29

               uv1 = up  + (upi - up)* xx
               uv2 = upj + (upij - upj)* xx
               uv  = uv1 + (uv2 - uv1)* yy

               vv1 = vp  + (vpi - vp)* xx
               vv2 = vpj + (vpij - vpj)* xx
               vv  = vv1 + (vv2 - vv1)* yy

               auv = ABS(uv) 
               avv = ABS(vv) 

               IF (auv .LT. 0.001 .AND. avv .LT. 0.001) GO TO 29
               IF (auv .GT. avv) THEN
                  uv = uv* fact/ auv
                  vv = vv* fact/ auv
               ELSE
                  uv = uv* fact/ avv
                  vv = vv* fact/ avv
               ENDIF


C From re_orient.F
C will we need COS(latitude) corrections ? ( the true dx may be dx*COS(lat) )
C compute COS(lat) correction at this latitude

           IF ( need_cos ) THEN

               IF (JTYPE.EQ.-1)THEN
                  xx = dx* x + xtm
                  yy = dy* y + ytm
               ELSE
                  mx = INT(x)
                  my = INT(y) 
                  xinterp = px(mx+1)
                  yinterp = py(my+1)
                  call trans(0,xinterp,yinterp,xx,yy)
              ENDIF

              cos_factor = COS( deg2rad * yy )
              IF (trnspz) cos_factor = 1./COS( deg2rad * xx )
              IF ( cos_factor .LT. 1.E-6 ) cos_factor = 1.E-6 !arbitrary fix /0
           ELSE
              cos_factor = 1.0
           ENDIF

C  travelling the other direction, so subtract 

               x = x - uv/cos_factor
               y = y - vv/cos_factor

C  Correction for irregular grids and rectangular regions.
C  and for latitude

               IF (jtype .NE. -1) THEN	! get local dx, dy
                  mx = MAX(1., x + 1)
                  my = MAX(1., y + 1)
                  IF (mx .LT. nxs) THEN
                     dx = px(mx+1) - px(mx)
                  ELSE 
                     dx = px(mx) - px(mx-1)
                  ENDIF
                  IF (my .LT. nys) THEN
                     dy = py(my+1) - py(my)
                  ELSE 
                     dy = py(my) - py(my-1)
                  ENDIF
               ENDIF

C  travelling the other direction, so subtract 
 
               IF (.NOT. curvilinear) THEN
                  IF (.NOT. trnspz) THEN
                     x = x - uv/ (dx*cos_factor/dy)
                     y = y - vv/ (dy/(dx*cos_factor))
                  ELSE
                     x = x - uv/ (dx/(dy*cos_factor))
                     y = y - vv/ (dy*cos_factor/dx)
                  ENDIF
               ENDIF

               ii1 = INT(x*rscl)
               ij1 = INT(y*rscl)
               ii1 = ij1*iss + ii1

               ii1index = ii1 + 1

               IF (ii1 .LT. 0  .OR.  ii1 .GE. siz) GO TO 29
               IF (startflag(ii1index) .EQ. 1.) GO TO 29

               IF (ii1 .NE. iisav .AND. iisav .GT. 0)startflag(iisav) =1. 

               IF (ii1index .EQ. iisav) THEN
                  iacc = iacc + 1
               ELSE
                  iacc = 0
               ENDIF

               IF (iacc .GT. 10) GO TO 29
               iisav = ii1index

               IF(JTYPE.EQ.-1)THEN      
                  xx = dx* x + xtm
                  yy = dy* y + ytm

               ELSE
                  mx = INT(x)
                  my = INT(y) 
                  xup = x - FLOAT(mx)
                  yup = y - FLOAT(my)
                  IF (mx+2 .LE. nxs) THEN
                     xinterp = px(mx+1) + xup* (px(mx+2) - px(mx+1))  
                  ELSE
                     xinterp = px(nxs)
                  ENDIF

                  IF (my+2 .LE. nys) THEN
                     yinterp = py(my+1) + yup* (py(my+2) - py(my+1))
                  ELSE
                     yinterp = py(nys)
                  ENDIF

                  call trans(0,xinterp,yinterp,xx,yy)
               ENDIF


               IF (icol .GT. -1) THEN
                  IF (bflg) THEN

                     IF (curvilinear) THEN
                        x_curv = xold
                        y_curv = xold
                        CALL curv_coord (x_curv, y_curv, 1, 1.0, 1.0, 
     .                      status) ! xform the pts
                        xxc=x_curv
                        yyc=y_curv
                        CALL plot (xxc,yyc,0,0)
                     ELSE
                        CALL plot(xold,yold,0,0)
                     ENDIF
                     bflg = .FALSE.
                  ENDIF

                  IF (curvilinear) THEN
                     x_curv = xx
                     y_curv = yy
                     CALL curv_coord (x_curv, y_curv, 1, 1.0, 1.0, 
     .                   status) ! xform the pts
                     xxc=x_curv
                     yyc=y_curv
                     CALL plot (xxc,yyc,1,0)
                  ELSE
                     CALL plot(xx,yy,1,0)
         
                  ENDIF
               ELSE
                  bflg = .TRUE.
               ENDIF

               xold = xx
               yold = yy
               acnt = acnt + 1

C  Draw arrow head
               IF (acnt .GT. 20) THEN
                 vel = SQRT(upsv*upsv + vpsv*vpsv)* vfact
                 IF (icol .GT. -1) 
     .              CALL strmar(xx, yy, xxsv, yysv, vel, xf, yf)
                  acnt = 0
               ENDIF

               xxsv = xx 
               yysv = yy
               upsv = up
               vpsv = vp
            ENDDO
   29       CONTINUE

         ENDIF
         i2 = i2 + 1
         IF (i2 .EQ. iss) THEN
            i2 = 1
            j2 = j2 + 1
         ENDIF

      ENDDO

C  From vector
C
C	DRAW SCALE  make arrow scale at XT,YT
C

C  Make a label containing the scale factor. 
C  Save it in a symbol even if the label not drawn.

	CALL COLOR(IPEN(0))
	WRITE (VECLAB,66) 1./vfact
  66    FORMAT ('flow arrow scale', 1pg8.2)

C   Save the value in a symbol PPL_VECLEN
	symbuff = 'PPL_VECLEN'
	CALL GETSYM( symbuff, buff, i2, i1 )
	IF ( i1 .EQ. 0 ) CALL DELSYM(symbuff,i1)
	slen = TM_LENSTR1(veclab)
	CALL PUTSYM( symbuff, veclab, slen, i2)

	IF(.NOT. VKEY)THEN
	    CALL COLOR(IPEN(0))
	    RETURN
	ENDIF
	CALL WINDOW(0.,0.,0.,0.)
     
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
        
        XX1 = XT + 0.1/XFUSER  
	YY1 = YT 

C  Make a sample line and label it with the scale factor. 

        CALL PLOT (xx1,yy1,0,0)
        CALL PLOT (xt,yy1,1,0)

C	DRAW LABEL

	CALL COLOR(IPEN(0))
	ILN=LNBLK(VECLAB,30)
c	XL = XX1 + 2.* hlabs/XF+0.2/xf
	XL = XX1 + .125/XF
	YL = YY1 - HLABS*0.5/YF
	CALL SYMBEL(XL,YL,0.,HLABS,ILN,VECLAB)

3333	IF(IWIND.GT.0)CALL WINDOW(XLO,YLO,XHI,YHI)

      RETURN
* error exit
 5100 curv_missing = .TRUE.
      RETURN
      END
