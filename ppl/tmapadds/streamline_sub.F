      SUBROUTINE streamline_sub (u, v, startflag, siz, den)


C**
C**    @(#)streamline_sub.F	    10/03/00
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

C	ARRAY X HOLD 2 GRIDS, U(X) IN THE FIRST
C	NX BY NY GRID AND V(Y) IN THE SECOND
C       startflag is a work array.
C
      INTEGER siz, den
      INTEGER startflag(*)
      REAL u(nx,ny),v(nx,ny)
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

#ifdef unix
        include         'xinterrupt.cmn' ! 1/89 *sh*
        include         'ppl_in_ferret.cmn'
#else
        INCLUDE         'TMAP_PPLV11INC:XINTERRUPT.CMN' ! 1/89 *sh*
        INCLUDE         'TMAP_PPLV11INC:PPL_IN_FERRET.CMN'
#endif


*******************************************************************************




C  From from GrADS module gxstrm.c

C void gxstrm (float *u, float *v, float *c, int is, int js,
C   float uund, float vund, float cund, int flag, float *shdlvs,
C   int *shdcls, int shdcnt, int den)

      INTEGER is, js

      REAL up, vp 
      REAL x,y,xx,yy,uv1,uv2,uv,vv1,vv2,vv,auv,avv,xsav,ysav,xold,yold
      INTEGER i,ii,jj,ipen,ii1,ij1,i2,j2,ipt,acnt,icol,scol,dis
      INTEGER iindex, jjndex, ii1index
      INTEGER iacc,iisav,iscl,imn,imx,jmn,jmx,iz,jz,iss,jss,bflg
      REAL fact,rscl,xxsv,yysv
      REAL xtm, ytm, xtx, ytx
      REAL afact, upsv, vpsv


C     Curvilinear coordinates declarations
      INTEGER status
      REAL x_curv, y_curv

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

C  Adjust arrow lengths so they fit the plot

      afact = (float(nx) + float(ny) )/ 100.

C  Get max velocity for scaling the arrows

      velmx = 0.
      DO i = 1, nx
         DO j = 1, ny 
            up = u(i,j)
            vp = v(i,j)
            IF (ZLE .AND. (up .LE. CMZLE .OR. vp .LE. CMZLE)) GO TO 89
            IF (ZEQ .AND. (up .EQ. CMZEQ .OR. vp .EQ. CMZEQ)) GO TO 89
            IF (ZGE .AND. (up .GE. CMZGE .OR. vp .GE. CMZGE)) GO TO 89

            vel = sqrt(up*up + vp*vp)
            velmx = max(velmx, vel)

  89        CONTINUE
         ENDDO
      ENDDO 

C  Initialize the flag grid

      DO i = 1, siz
         startflag(i) = 0
      ENDDO

C   Loop through flag grid to look for start of streamlines.  
C     To start requires no streams drawn within surrounding 
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
              iacc = iacc + startflag(ipt)
              ipt = ipt + 1
            ENDDO
         ENDDO
         IF (iacc .EQ. 0) THEN
            x = REAL((i2-1))/rscl
            y = REAL((j2-1))/rscl
            xsav = x
            ysav = y

            xx = dx* x + xtm
            yy = dy* y + ytm
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
            bflg = 0

            DO WHILE (x .GT. 0.0 .AND. x .LT. REAL(is-1) .AND. 
     .                y .GT. 0.0 .AND. y .LT. REAL(js-1)) 
               ii = INT(x) 
               jj = INT(y)
               xx = x - REAL(ii)
               yy = y - REAL(jj)
               iindex = ii + 1		! ACM: indices must start at 1 not 0.
               jjndex = jj + 1

               up = u(iindex,jjndex)
               upi = u(iindex+1,jjndex)
               upj = u(iindex,jjndex+1)
               upij = u(iindex+1,jjndex+1)

               vp = v(iindex,jjndex)
               vpi = v(iindex+1,jjndex)
               vpj = v(iindex,jjndex+1)
               vpij = v(iindex+1,jjndex+1)

               IF (ZLE .AND.
     .               (up .LE. CMZLE .OR. upi .LE. CMZLE .OR.
     .                upj.LE. CMZLE .OR. upij.LE. CMCLE .OR.          
     .                vp .LE. CMZLE .OR. vpi .LE. CMCLE .OR.
     .                vpj.LE. CMZLE .OR. vpij.LE. CMCLE)) GO TO 19

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

 
               uv1 = up + (upi - up)* xx
               uv2 = upj + (upij - upj)* xx
               uv = uv1 + (uv2 - uv1)* yy

               vv1 = vp + (vpi - vp)* xx
               vv2 = vpj + (vpij - vpj)* xx
               vv = vv1 + (vv2-vv1)*yy

               auv = ABS(uv) 
               avv = ABS(vv)
               IF (auv .LT. 0.1 .AND. avv .LT. 0.1) GO TO 19
               IF (auv .GT. avv) THEN
                  vv = vv*fact/auv
                  uv = uv*fact/auv
               ELSE
                  uv = uv*fact/avv
                  vv = vv*fact/avv
               ENDIF

               x = x + uv
               y = y + vv

               ii1 = INT(x*rscl)
               ij1 = INT(y*rscl)
               ii1 = ij1*iss + ii1
               ii1index = ii1 + 1

               IF (ii1 .LT. 0  .OR.  ii1 .GE. siz) GO TO 19
               IF (startflag(ii1index) .EQ. 1) GO TO 19
               IF (ii1index .NE. iisav .AND. iisav.GT.-1) 
     .              startflag(iisav) = 1
               IF (ii1index .EQ. iisav) THEN
                  iacc = iacc + 1
               ELSE
                  iacc = 0
               ENDIF

               IF (iacc .GT. 10) GO TO 19

               iisav = ii1index
               xx = dx* x + xtm
               yy = dy* y + ytm

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

                     bflg=0
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
                  bflg = 1
               ENDIF

               xold = xx
               yold = yy
               acnt = acnt + 1

C  Draw arrow heads

               IF (acnt .GT. 20) THEN
                 vel = sqrt(upsv*upsv + vpsv*vpsv)/ velmx
                 IF (icol .GT. -1) 
     .               CALL strmar (xxsv, yysv, xx, yy, afact, vel)
                  acnt = 0
               ENDIF

               xxsv = xx 
               yysv = yy
               upsv = up
               vpsv = vp
            ENDDO

   19       CONTINUE
            bflg = 0
            x = xsav 
            y = ysav
            xx = dx* x + xtm
            yy = dy* y + ytm
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

               up = u(iindex,jjndex)
               upi = u(iindex+1,jjndex)
               upj = u(iindex,jjndex+1)
               upij = u(iindex+1,jjndex+1)

               vp = v(iindex,jjndex)
               vpi = v(iindex+1,jjndex)
               vpj = v(iindex,jjndex+1)
               vpij = v(iindex+1,jjndex+1)

               IF (ZLE .AND.
     .               (up .LE. CMZLE .OR. upi .LE. CMZLE .OR.
     .                upj.LE. CMZLE .OR. upij.LE. CMCLE .OR.          
     .                vp .LE. CMZLE .OR. vpi .LE. CMCLE .OR.
     .                vpj.LE. CMZLE .OR. vpij.LE. CMCLE)) GO TO 29

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

               uv1 = up + (upi - up)* xx
               uv2 = upj + (upij - upj)* xx
               uv = uv1 + (uv2 - uv1)* yy

               vv1 = vp + (vpi - vp)* xx
               vv2 = vpj + (vpij - vpj)* xx
               vv = vv1 + (vv2 - vv1)* yy

               auv = ABS(uv) 
               avv = ABS(vv) 

               IF (auv .LT. 0.1 .AND. avv .LT. 0.1) GO TO 29
               IF (auv .GT. avv) THEN
                  vv = vv* fact/ auv
                  uv = uv* fact/ auv
               ELSE
                  uv = uv* fact/ avv
                  vv = vv* fact/ avv
               ENDIF

               x = x - uv
               y = y - vv

               ii1 = INT(x*rscl)
               ij1 = INT(y*rscl)
               ii1 = ij1*iss + ii1

               ii1index = ii1 + 1

               IF (ii1 .LT. 0  .OR.  ii1 .GE. siz) GO TO 29
               IF (startflag(ii1index) .EQ. 1) GO TO 29

               IF (ii1 .NE. iisav .AND. iisav .GT. 0)
     .             startflag(iisav) = 1  

               IF (ii1index .EQ. iisav) THEN
                  iacc = iacc + 1
               ELSE
                  iacc = 0
               ENDIF

               IF (iacc .GT. 10) GO TO 29
               iisav = ii1index

               xx = dx* x + xtm
               yy = dy* y + ytm

               IF (icol .GT. -1) THEN
                  IF (bflg .EQ. 1) THEN

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
                     bflg=0
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
                  bflg = 1
               ENDIF

               xold = xx
               yold = yy
               acnt = acnt + 1

C  Draw arrow head
               IF (acnt .GT. 20) THEN
                 vel = sqrt(upsv*upsv + vpsv*vpsv)/ velmx
                 IF (icol .GT. -1) 
     .                 CALL strmar(xx, yy, xxsv, yysv, afact, vel)
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
      CALL COLOR(IPEN(0))

      RETURN
      END

C  arrows 

      subroutine strmar (xx1, yy1, xx2, yy2, afact, vel) 
      PARAMETER (a160 = 160.0*3.1416/180.0)
      REAL afact, vel, dir, xx1, xx2, yy1, yy2

#ifdef unix
        include 'vector_inc.decl'
        include 'VECTOR.INC'
        include 'xcurvilinear.cmn'
#else
        INCLUDE 'tmap_pplv11inc:tmap_VECTOR.INC'
#endif

C     Curvilinear coordinates declarations
      INTEGER status
      REAL x_curv, y_curv

      arrw = afact* vel* 2.

      dir = atan2(yy2-yy1,xx2-xx1)

      IF (curvilinear) then

          x_curv = xx2
          y_curv = yy2
          CALL curv_coord (x_curv, y_curv, 1, 1.0, 1.0, status)
          xxc=x_curv
          yyc=y_curv
          CALL plot (xxc,yyc,0,0)

          x_curv = xx2+arrw*cos(dir+a160)
          y_curv = yy2+arrw*sin(dir+a160)
          CALL curv_coord (x_curv, y_curv, 1, 1.0, 1.0, status)
          xxd=x_curv
          yyd=y_curv
          CALL plot (xxd,yyd,1,0)
          CALL plot (xxc,yyc,0,0)


          x_curv = xx2+arrw*cos(dir-a160)
          y_curv = yy2+arrw*sin(dir-a160)
          CALL curv_coord (x_curv, y_curv, 1, 1.0, 1.0, status)
          xxe=x_curv
          yye=y_curv
          CALL plot (xxe,yye,1,0)

          CALL plot (xxc,yyc,0,0)

      ELSE
         dir = atan2(yy2-yy1,xx2-xx1)

         CALL plot (xx2,yy2,0,0)
         CALL plot (xx2+arrw*cos(dir+a160),yy2+arrw*sin(dir+a160),1,0)
         CALL plot (xx2,yy2,0,0)
         CALL plot (xx2+arrw*cos(dir-a160),yy2+arrw*sin(dir-a160),1,0)
         CALL plot (xx2,yy2,0,0)
      ENDIF

      RETURN
      END
