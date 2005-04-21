      subroutine drawpoly(x, y, z, nsides)
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
c:    
c:    Output polygon (rectangle or triangle) for contour display.
c:    
* V581 ACM 3/05 removed to its own file from arfill
* v581 *acm* 3/05 open levels (-INF) (INF)

      real x(*), y(*), z(*)
      integer i,j,npoly,nsides
      real xpoly(20), ypoly(20),zmin,zmax,f
c
        include         'miss_inc.decl'
        include         'MISS.INC'
        include         'shade_vars.cmn'
        include         'xcurvilinear.cmn'

	include		'parampl5_dat.decl'
	include		'PARAMPL5.DAT'
	include		'cont_inc.decl'
	include		'CONT.INC'

      zmin=  1e30
      zmax= -1e30
      
      lev1 = 1
      levn = nlev2
      IF (neginf) lev1 = 2
      IF (posinf) levn = nlev2 - 1

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
c      do 100 i = lev1, levn-1
         IF (i.EQ.lev1 .AND. neginf .AND.
     .       (zmin .LT. zlev(lev1) .AND. zmax.LT.zlev(lev1)) ) THEN

*        ENTIRE CELL IS BELOW MIN (-INF)
            if ((i .ne. ilast .and. .not. buffer_empty) 
     .           .or. num_buffered .ge. buffer_max) then

               call flush_buffered_fillpoly
            end if

            ndx = lev1-2
            call buffer_fillpoly (x, y, nsides, ndx)
            return

         ELSE IF (i.EQ.levn-1 .AND. posinf .AND.
     .       (zmin .GT. zlev(levn) .AND. zmax.GT.zlev(levn)) ) THEN

*        ENTIRE CELL IS ABOVE MAX (INF)
            if ((i .ne. ilast .and. .not. buffer_empty) 
     .           .or. num_buffered .ge. buffer_max) then

               call flush_buffered_fillpoly
            end if

            ndx = nlev2-2
            call buffer_fillpoly (x, y, nsides, ndx)
            return
         ELSE

*        ENTIRE CELL IS OUTSIDE THIS LEVEL
            if ((zmin.gt.zlev(i+1)) .or. (zmax.lt.zlev(i))) goto 100

*        ENTIRE CELL IS A SINGLE COLOR
            if ((zmin.ge.zlev(i)) .and. (zmax.le.zlev(i+1))) then

               if ((i .ne. ilast .and. .not. buffer_empty) 
     .           .or. num_buffered .ge. buffer_max) then

                  call flush_buffered_fillpoly
               end if

               ndx = i-1
               call buffer_fillpoly (x, y, nsides, ndx)
               return
            endif
         ENDIF
*        CELL IS MORE THAN ONE COLOR. FLUSH BUFFER AND FILL IN THIS CELL.
         if (.not. buffer_empty) call flush_buffered_fillpoly

         x(nsides+1)=x(1)
         y(nsides+1)=y(1)
         z(nsides+1)=z(1)
         npoly=0
         ndx = i-1
c
         do 80 j=1,nsides

            if (z(j).lt.zlev(i)) then
            IF (i .le. lev1) THEN
              nothing = 0
            ENDIF
 
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
            IF (i .ge. levn) THEN
              nothing = 0
            ENDIF
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

         call fillpoly(xpoly, ypoly, npoly, ndx)

 100   continue

      return 
      end
