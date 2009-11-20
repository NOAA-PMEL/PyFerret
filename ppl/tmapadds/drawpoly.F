      subroutine drawpoly(x, y, z, zlev, nlev, nsides, xadd, lmin, lmax)
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
c:    Extracted from arfill.F
c:    Output polygon (rectangle or triangle) for contour display.
c:
*     v5.81 *acm* 4/05 Changes for FILL/MOD. Draw replications for curvilinear datasets
*     v6.4  *acm*11/09 For curvi plots, dont buffer polygons of a single color. This 
*                      resulted in drawing rectangles across what should be curvi portions
*                      of the grid (bug 1505).

      real x(*), y(*), z(*)
      integer i,j,npoly,nlev,nsides 
      real xadd, lmin, lmax
      real zlev(nlev)
      real xpoly(20), ypoly(20),zmin,zmax,f
c
        include         'miss_inc.decl'
        include         'MISS.INC'
        include         'shade_vars.cmn'
        include         'xcurvilinear.cmn'

      zmin=  1e30
      zmax= -1e30

      do 10 i = 1,nsides
         if ((zle .and. z(i) .le. cmzle)  .or.
     .        (zeq .and. z(i) .eq. cmzeq)  .or.
     .        (zge .and. z(i) .ge. cmzge)) then
            
            if (.not. buffer_empty) call flush_buffered_fillpoly (xadd,
     .                    lmin, lmax)
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
     .           .or. num_buffered .ge. buffer_max
     .           .or. curvilinear) then

               call flush_buffered_fillpoly (xadd, lmin, lmax)
            end if

            call buffer_fillpoly (x, y, nsides, i-1)
            return
         endif

*        CELL IS MORE THAN ONE COLOR. FLUSH BUFFER AND FILL IN THIS CELL.
         if (.not. buffer_empty) call flush_buffered_fillpoly (xadd,
     .                     lmin, lmax)

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

         call fillpoly(xpoly, ypoly, npoly, i-1, xadd, lmin, lmax)

 100   continue

      return 
      end
