      subroutine arfill(z,x,y,nx,ny,zlev,nlev)
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
*     The variable nlev2 is passed here as nlev *jd*
*
C** 
C**    @(#)arfill.F	1.6    2/6/91
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C     lots of places ---> PPLMOD_PPLCMD ---> PPLMOD_PLTIT 
C              ---> PPLMOD_PLOTZ ---> PPLMOD_ARFILL
C
C     This subroutine ...
C
C     z      - actual 2-D array of data values to be contoured
C     x      - array of grid points along the X axis
C     y      - array of grid points along the Y axis
C     nx     - number of X grid points
C     ny     - number of Y grid points
C     zlev   - array of contour level values (lo val --> hi val)
C     nlev   - number of levels to use
C
C
*	MODS JD 1.13.92 To fit into TMAP PPL+
*	Mod  *jd* 10.26.93 eliminate private item for meta_shade_set in xgks
*			 - not replaced with appropriate code yet
* 	Mod *jd * 1.30.96  Fix bug in fill --- missing value flag is 
*				unrecognized.
C     V4.50 *jc* 5.97 Major cleanup and commenting
C     V4.50 *jc* 5.97 Add curvilinear coordinates code
C     v5.00 *jd* 2.99 Add fill optimization 

#ifdef unix
	include 	'pplv11inc/ARFILL.INC'
        include         'pplv11inc/GKSCM1.INC'
        include         'tmap_pplv11inc/gkscm2.cmn'
        include         'tmap_pplv11inc/miss_inc.decl'
        include         'pplv11inc/MISS.INC'
        include         'ferret_cmn/xcurvilinear.cmn'
#else
        INCLUDE         'tmap_pplv11inc:tmap_miss.inc'
	INCLUDE 	'pplinc:ARFILL.INC'
        INCLUDE         'tmap_pplv11inc:tmap_gkscm1.inc'
        INCLUDE         'tmap_pplv11inc:tmap_gkscm2.inc'
#endif

#ifdef unix
        include         'tmap_pplv11inc/xinterrupt.cmn' ! 1/89 *sh*
        include         'tmap_pplv11inc/ppl_in_ferret.cmn'
        include         'tmap_pplv11inc/shade_vars.cmn'
#else
        INCLUDE         'TMAP_PPLV11INC:XINTERRUPT.CMN' ! 1/89 *sh*
        INCLUDE         'TMAP_PPLV11INC:PPL_IN_FERRET.CMN'
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
#endif

        integer nx,ny,nlev
        real z(nx,ny),x(nx),y(ny),zlev(nlev)

        real xt(5),yt(5),zt(5),x_extra,y_extra
        integer i,j,ii,jj,nnx,nny


C     *jd* mod to allow filling single level fields
        if (nlev .le. 1) return

	shd_levels = nlev - 1

C     DETERMINE PATTERNS OR COLORS TO SET BUNDLED FILL AREA REPRESENTATIONS
        call set_fill_area_rep(wsid,wstype)

C     WRITE OUT SHD_LEVELS & SHADE SETTINGS TO METAFILE (4.91)
#ifndef xgks
        if (meta_actv) call meta_shade_set
#endif

C     The FILL command does not deal with curvilinear modulo axes properly.
C     The problem is seen in polar projections where the longitude 0E and
C     longitude 360E are not connected: a small pie wedge is left blank.
C
C     To remedy this, we should test for modulo_curvilinear axes and
C     replicate the appropriate row/column in the z array before 
C     breaking the data up into small triangles (or squares) which
C     are then fed to drawpoly().
C
C     If an axis is modulo_curvi we will increase nnx(nny) by one and
C     create an extra point for the end of that axis
C     This code works properly for regularly gridded axes.  For irregular
C     axes it is as good a guess as any.

        if (curvilinear .and. dim1_modulo) then
           x_extra = x(nx) + (x(nx)-x(nx-1))
           nnx = nx+1
        else
           nnx = nx
        endif
 
        if (curvilinear .and. dim2_modulo) then
           y_extra = y(ny) + (y(ny)-y(ny-1))
           nny = ny+1
        else
           nny = ny
        endif


C     Divide the region into rectangles or triangles and feed these
C     small elements to drawpoly()

        buffer_empty = .true.

        do 200 j=1,nny-1
           jj = j
           do 100 i=1,nnx-1
              ii = i

              IF (interrupted) RETURN
     
C     For curvilinear modulo axes:
C     Replicate the first point in a column/row if necessary
              if (i.eq.nx) ii = 1
              if (j.eq.ny) jj = 1

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
                 else
                    xt(2)=x(i+1)
                 end if
                 yt(2)=yt(1)
                 
                 zt(3)=z(ii+1,jj+1)
                 if (i.eq.nx) then
                    xt(3) = x_extra
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

                 call drawpoly(xt,yt,zt,zlev,nlev,4)

              else
c     
c     do upper triangle
c     
                 upperflg = .true.

                 zt(2)=z(ii+1,jj+1)
                 if (i.eq.nx) then
                    xt(2) = x_extra
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
     
                 call drawpoly(xt,yt,zt,zlev,nlev,3)
c     
c     do lower triangle
c     
                 upperflg = .false.

                 zt(3)=z(ii+1,jj)
                 xt(3)=xt(2)
                 yt(3)=yt(1)
     
                 call drawpoly(xt,yt,zt,zlev,nlev,3)

              endif

 100        continue
            if (.not. buffer_empty) call flush_buffered_fillpoly
 200     continue
         
C     
C     Generate a color/gray scale key.
C     
        if (do_key .AND. .NOT.interrupted) call key ! 1/89 - *sh*

        return
        end


C======================================================================

c:    
c:    Output polygon (rectangle or triangle) for contour display.
c:    
      subroutine drawpoly(x, y, z, zlev, nlev, nsides)
      real x(*), y(*), z(*)
      integer i,j,npoly,nlev,nsides
      real zlev(nlev)
      real xpoly(20), ypoly(20),zmin,zmax,f
c

#ifdef unix
        include         'tmap_pplv11inc/miss_inc.decl'
        include         'pplv11inc/MISS.INC'
        include         'tmap_pplv11inc/shade_vars.cmn'
        include         'ferret_cmn/xcurvilinear.cmn'
#else
        INCLUDE         'tmap_pplv11inc:tmap_miss.inc'
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
#endif

      zmin=  1e30
      zmax= -1e30

      do 10 i = 1,nsides
         if ((zle .and. z(i) .le. cmzle)  .or.
     .        (zeq .and. z(i) .eq. cmzeq)  .or.
     .        (zge .and. z(i) .ge. cmzge)) then
            
            if (.not. buffer_empty) call flush_buffered_fillpoly
            return
         end if
         
         zmin=amin1(zmin,z(i))
         zmax=amax1(zmax,z(i))
 10   continue
      
      do 100 i = 1, nlev-1

*        ENTIRE CELL IS OUTSIDE PRESET LEVELS
         if ((zmin.gt.zlev(i+1)) .or. (zmax.lt.zlev(i))) goto 100

*        ENTIRE CELL IS A SINGLE COLOR
         if ((zmin.ge.zlev(i)) .and. (zmax.le.zlev(i+1))) then

            if ((i .ne. ilast .and. .not. buffer_empty) 
     .           .or. num_buffered .ge. buffer_max) then

               call flush_buffered_fillpoly
            end if

            call buffer_fillpoly (x, y, nsides, i-1)
            return
         endif

*        CELL IS MORE THAN ONE COLOR. FLUSH BUFFER AND FILL IN THIS CELL.
         if (.not. buffer_empty) call flush_buffered_fillpoly

         x(nsides+1)=x(1)
         y(nsides+1)=y(1)
         z(nsides+1)=z(1)
         npoly=0
c
         do 80 j=1,nsides

            if (z(j).lt.zlev(i)) then
c
c z(j) is below
c
               if (z(j+1).gt.zlev(i+1)) then
c
c z(j+1) is above
c
                  npoly=npoly+1
                  f=(z(j)-zlev(i))/(z(j)-z(j+1))
                  xpoly(npoly)=x(j) - f*(x(j)-x(j+1))
                  ypoly(npoly)=y(j) - f*(y(j)-y(j+1))
c
                  npoly=npoly+1
                  f=(z(j)-zlev(i+1))/(z(j)-z(j+1))
                  xpoly(npoly)=x(j) - f*(x(j)-x(j+1))
                  ypoly(npoly)=y(j) - f*(y(j)-y(j+1))

               else if ((z(j+1).ge.zlev(i)) .and. 
     *                 (z(j+1).le.zlev(i+1))) then
c
c z(j+1) is inside
c
                  npoly=npoly+1
                  f=(z(j)-zlev(i))/(z(j)-z(j+1))
                  xpoly(npoly)=x(j) - f*(x(j)-x(j+1))
                  ypoly(npoly)=y(j) - f*(y(j)-y(j+1))
c
                  npoly=npoly+1
                  xpoly(npoly)=x(j+1)
                  ypoly(npoly)=y(j+1)

               endif

            else if (z(j).gt.zlev(i+1)) then
c
c z(j) is above
c
               if (z(j+1).lt.zlev(i)) then
c
c z(j+1) is below
c
                  npoly=npoly+1
                  f=(z(j)-zlev(i+1))/(z(j)-z(j+1))
                  xpoly(npoly)=x(j) - f*(x(j)-x(j+1))
                  ypoly(npoly)=y(j) - f*(y(j)-y(j+1))
c
                  npoly=npoly+1
                  f=(z(j)-zlev(i))/(z(j)-z(j+1))
                  xpoly(npoly)=x(j) - f*(x(j)-x(j+1))
                  ypoly(npoly)=y(j) - f*(y(j)-y(j+1))

               else if ((z(j+1).ge.zlev(i)) .and. 
     *                 (z(j+1).le.zlev(i+1))) then
c
c z(j+1) is inside
c
                  npoly=npoly+1
                  f=(z(j)-zlev(i+1))/(z(j)-z(j+1))
                  xpoly(npoly)=x(j) - f*(x(j)-x(j+1))
                  ypoly(npoly)=y(j) - f*(y(j)-y(j+1))
c
                  npoly=npoly+1
                  xpoly(npoly)=x(j+1)
                  ypoly(npoly)=y(j+1)

               endif

            else
c
c z(j) is inside
c
               if (z(j+1).gt.zlev(i+1)) then
c
c z(j+1) is above
c
                  npoly=npoly+1
                  f=(z(j)-zlev(i+1))/(z(j)-z(j+1))
                  xpoly(npoly)=x(j) - f*(x(j)-x(j+1))
                  ypoly(npoly)=y(j) - f*(y(j)-y(j+1))

               else if(z(j+1).lt.zlev(i))then
c
c z(j+1) is below
c
                  npoly=npoly+1
                  f=(z(j)-zlev(i))/(z(j)-z(j+1))
                  xpoly(npoly)=x(j) - f*(x(j)-x(j+1))
                  ypoly(npoly)=y(j) - f*(y(j)-y(j+1))

               else
c
c z(j+1) is inside
c
                  npoly=npoly+1
                  xpoly(npoly)=x(j+1)
                  ypoly(npoly)=y(j+1)

               endif

            endif

 80      continue

         call fillpoly(xpoly, ypoly, npoly, i-1)

 100   continue

      return 
      end


C======================================================================

      subroutine buffer_fillpoly (xarg,yarg,npts,index)

#ifdef unix
        include         'tmap_pplv11inc/xinterrupt.cmn' ! 1/89 *sh*
        include         'tmap_pplv11inc/ppl_in_ferret.cmn'
        include         'tmap_pplv11inc/shade_vars.cmn'
#else
        INCLUDE         'TMAP_PPLV11INC:XINTERRUPT.CMN' ! 1/89 *sh*
        INCLUDE         'TMAP_PPLV11INC:PPL_IN_FERRET.CMN'
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
#endif

*     J Davison 2.4.99

*     BUFFER_FILLPOLY() called when entire grid cell is a single color; when
*     color changes construct new polygon from individual cells and fill
*     it in FLUSH_BUFFERED_FILLPOLY().

      real xarg(*), yarg(*)
      integer npts, index, i

      buffer_empty = .false.
      num_buffered = num_buffered + 1

      if (npts .eq. 3 .and. num_buffered .eq. 1)
     .     upper_first = upperflg

      do 10 i = 1, npts
         xbuff(i,num_buffered) = xarg(i)         
         ybuff(i,num_buffered) = yarg(i)
 10   end do

      ilast = index + 1

      return
      end

C =========================================================================

      subroutine flush_buffered_fillpoly

#ifdef unix
        include         'tmap_pplv11inc/xinterrupt.cmn' ! 1/89 *sh*
        include         'tmap_pplv11inc/ppl_in_ferret.cmn'
        include         'tmap_pplv11inc/shade_vars.cmn'
	include 	'pplv11inc/ARFILL.INC'
        include         'ferret_cmn/xcurvilinear.cmn'
#else
        INCLUDE         'TMAP_PPLV11INC:XINTERRUPT.CMN' ! 1/89 *sh*
        INCLUDE         'TMAP_PPLV11INC:PPL_IN_FERRET.CMN'
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
	INCLUDE 	'pplinc:ARFILL.INC'
#endif
 
*     J Davison 2.4.99

*     BUFFER_FILLPOLY() called when entire grid cell is a single color; when
*     color changes construct new polygon from individual cells and fill
*     it in FLUSH_BUFFERED_FILLPOLY().

      real xarg(2*buffer_max+2), yarg(2*buffer_max+2)
      integer m, ndx, neo, noe
      logical even

      if (sqfflg) then
*        ********************* RECTANGLES ********************
         if (curvilinear) then

*           FIRST POINT IS LOWER LEFT CORNER OF FIRST RECTANGLE
            xarg(1) = xbuff(1,1)
            yarg(1) = ybuff(1,1)
           
*           FOLLOWED BY LOWER RIGHT CORNER OF ALL RECTANGLES
            do 10 m = 1, num_buffered
               xarg(m+1) = xbuff(2,m)
               yarg(m+1) = ybuff(2,m)
 10         end do
            
*           PICK THE UPPER RIGHT CORNER OF THE LAST RECTANGLE
            xarg(num_buffered+2) = xbuff(3,num_buffered)
            yarg(num_buffered+2) = ybuff(3,num_buffered)
 
*           THEN RETURN ALONG UPPER LEFT CORNER OF EACH RECTANGLE
            ndx = 2*num_buffered + 3

            do 20 m = num_buffered, 1, -1
               xarg(ndx-m) = xbuff(4,m)
               yarg(ndx-m) = ybuff(4,m)
 20         end do

            call fillpoly (xarg, yarg, 2*num_buffered+2, ilast-1)
          else
            xarg(1) = xbuff(1,1)
            yarg(1) = ybuff(1,1)

            xarg(2) = xbuff(2,num_buffered)
            yarg(2) = ybuff(2,num_buffered)

            xarg(3) = xbuff(3,num_buffered)
            yarg(3) = ybuff(3,num_buffered)

            xarg(4) = xbuff(4,1)
            yarg(4) = ybuff(4,1)

            call fillpoly (xarg, yarg, 4, ilast-1)
         end if
      else
*        ******************* TRIANGLES *******************
         
         if (num_buffered .eq. 1) then
            
            xarg(1) = xbuff(1,1)
            yarg(1) = ybuff(1,1)

            xarg(2) = xbuff(2,1)
            yarg(2) = ybuff(2,1)

            xarg(3) = xbuff(3,1)
            yarg(3) = ybuff(3,1)

            call fillpoly (xarg, yarg, 3, ilast-1)
            goto 1000
         end if
            
         if (mod(num_buffered,2) .eq. 0) then
            even = .true.
         else
            even = .false.
         end if
         
         if (curvilinear) then

*           INDICES DEPEND ON WHETHER THERE ARE AN EVEN OR ODD # OF PTS
            if (even) then
               neo = num_buffered
               noe = num_buffered - 1
            else
               neo = num_buffered - 1
               noe = num_buffered
            end if

*           FIRST POINT IS LEFT CORNER OF FIRST TRIANGLE
            xarg(1) = xbuff(1,1)
            yarg(1) = ybuff(1,1)
               
*           FOLLOWED BY RIGHT CORNER OF ALL TRIANGLES
            do 110 m = 2, neo, 2
               xarg(m/2+1) = xbuff(3,m)
               yarg(m/2+1) = ybuff(3,m)
 110        end do
               
*           PICK THE RIGHT CORNER OF THE LAST TRIANGLE
            xarg(neo/2+2) = xbuff(2,num_buffered)
            yarg(neo/2+2) = ybuff(2,num_buffered)
               
*           THEN RETURN ALONG LEFT CORNER OF EACH TRIANGLE
            ndx = 2*num_buffered + 5
            
            do 120 m = noe, 1, -2
               xarg((ndx-m)/2) = xbuff(3,m)
               yarg((ndx-m)/2) = ybuff(3,m)
 120        end do

            call fillpoly (xarg, yarg, num_buffered+2, ilast-1)
         else
            if (upper_first .and. even) then
               xarg(1) = xbuff(1,1)
               yarg(1) = ybuff(1,1)
               
               xarg(2) = xbuff(3,num_buffered)
               yarg(2) = ybuff(3,num_buffered)
               
               xarg(3) = xbuff(2,num_buffered)
               yarg(3) = ybuff(2,num_buffered)
               
               xarg(4) = xbuff(3,1)
               yarg(4) = ybuff(3,1)

               call fillpoly (xarg, yarg, 4, ilast-1)
            else if (upper_first .and. .not.even) then

               xarg(1) = xbuff(1,1)
               yarg(1) = ybuff(1,1)
               
               xarg(2) = xbuff(1,num_buffered)
               yarg(2) = ybuff(1,num_buffered)
               
               xarg(3) = xbuff(2,num_buffered)
               yarg(3) = ybuff(2,num_buffered)
               
               xarg(4) = xbuff(3,1)
               yarg(4) = ybuff(3,1)

               call fillpoly (xarg, yarg, 4, ilast-1)
            else if (.not.upper_first .and. even) then

               xarg(1) = xbuff(1,1)
               yarg(1) = ybuff(1,1)
               
               xarg(2) = xbuff(1,num_buffered)
               yarg(2) = ybuff(1,num_buffered)
               
               xarg(3) = xbuff(2,num_buffered)
               yarg(3) = ybuff(2,num_buffered)
               
               xarg(4) = xbuff(2,1)
               yarg(4) = ybuff(2,1)

               call fillpoly (xarg, yarg, 4, ilast-1)
            else if (.not.upper_first .and. .not.even) then

               xarg(1) = xbuff(1,1)
               yarg(1) = ybuff(1,1)
               
               xarg(2) = xbuff(3,num_buffered)
               yarg(2) = ybuff(3,num_buffered)
               
               xarg(3) = xbuff(2,num_buffered)
               yarg(3) = ybuff(2,num_buffered)
               
               xarg(4) = xbuff(2,1)
               yarg(4) = ybuff(2,1)

               call fillpoly (xarg, yarg, 4, ilast-1)
            end if
         end if
      end if
      
 1000 buffer_empty = .true.
      num_buffered = 0
      
      return
      end

           



