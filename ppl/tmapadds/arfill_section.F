      subroutine arfill_section(x, y, z, zlev, nx, ny, nlev, nnx, nny, 
     .       x_extra, y_extra, lmin, lmax, ilo, ihi, jlo, jhi, xadd, 
     .       status)

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
*     extracted from arfill.F, fill a section of a modulo (curvilinear) plot.
*
* ACM 4/13/05

      REAL z(nx,ny),x(nx),y(ny),zlev(nlev), x_extra, y_extra, 
     .     xadd, lmin, lmax
      INTEGER nx, ny, nlev, nnx, nny, ilo, ihi, jlo, jhi, status

      integer i, j, ii, jj
      real xt(5), yt(5), zt(5)

	include 	'ARFILL.INC'
        include         'GKSCM1.INC'
        include         'gkscm2.cmn'
        include         'miss_inc.decl'
        include         'MISS.INC'
        include         'xcurvilinear.cmn'
        include         'xinterrupt.cmn' ! 1/89 *sh*
        include         'ppl_in_ferret.cmn'
        include         'shade_vars.cmn'
        INCLUDE         'errmsg.parm'


*  check for consistency: data range 1:ny, index range jlo:jhi  
*  If inconsistent, use whole Y range given by ny

      IF (curvilinear) THEN
         IF (jlo .LT. 1 .OR. jhi .GT. nny+1) THEN  
            jlo = 1
            jtop = nny
         ELSE
            jtop = jhi 
            IF (jhi .EQ. nny+1) jtop = nny
         ENDIF
      ELSE
         jlo = 1
         jtop = nny
      ENDIF

*  Likewise for X, if this is not an x modulo plot

      IF (curvilinear .AND. (.NOT. xmodulo) ) THEN
         IF (ilo .LT. 1 .OR. ihi .GT. nnx+1) THEN  
            ilo = 1
            itop = nnx
         ELSE
            itop = ihi 
            IF (ihi .EQ. nnx+1) itop = nnx
         ENDIF
      ELSE
         ilo = 1
         itop = nnx
      ENDIF


C     Divide the region into rectangles or triangles and feed these
C     small elements to drawpoly()

      buffer_empty = .true.

      do 200 j=jlo,jtop-1
         jj = j
         do 100 i=ilo,itop-1
            ii = i

            IF (interrupted) THEN
              status = ferr_interrupt
              RETURN
            ENDIF
     
C     For curvilinear modulo axes:
C     Replicate the first point in a column/row if necessary
            if (i.eq.nnx) ii = 1
            if (j.eq.nny) jj = 1

            zt(1)=z(ii,jj)
            xt(1) = x(i)
            yt(1) = y(j)

            if (sqfflg) then
c     
c     do a square
c     
               zt(2)=z(ii+1,jj)
               if (i.eq.nx) then
                  xt(2)=x_extra
                  zt(2)=z(ilo,jj)
               else
                  xt(2)=x(i+1)
               end if
               yt(2)=yt(1)
              
               zt(3)=z(ii+1,jj+1)
               if (i.eq.nx) then
                  xt(3)=x_extra
                  zt(3)=z(ilo,jj+1)
               else
                  xt(3)=x(i+1)
               end if

               if (j.eq.ny) then
                  yt(3)=y_extra
               else
                  yt(3)=y(j+1)
               endif
               
               zt(4)=z(ii,jj+1)
               xt(4)=xt(1)
               yt(4)=yt(3) 

               call drawpoly(xt,yt,zt,zlev,nlev,4,xadd,lmin,lmax)

            else
c     
c     do upper triangle
c     
              upperflg = .true.

              zt(2)=z(ii+1,jj+1)
              if (i.eq.nx) then
                 xt(2) = x_extra
                 zt(2) = z(ilo,jj+1)
              else
                 xt(2)=x(i+1)
              end if
              if (j.eq.ny) then
                 yt(2)=y_extra
              else
                 yt(2)=y(j+1)
              end if

              zt(3)=z(ii,jj+1)
              xt(3)=xt(1)
              yt(3)=yt(2)
     
              call drawpoly(xt,yt,zt,zlev,nlev,3,xadd,lmin,lmax)
c     
c     do lower triangle
c     
              upperflg = .false.

              zt(3)=z(ii+1,jj)
              xt(3)=xt(2)
              yt(3)=yt(1)
     
           call drawpoly(xt,yt,zt,zlev,nlev,3,xadd,lmin,lmax)

         endif

 100     continue
         if (.not. buffer_empty) call flush_buffered_fillpoly(xadd,
     .                     lmin, lmax)
 200  continue
      
      return
      end
