      subroutine arfill(z,x,y,nx,ny,zlev,nlev)
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


C     Divide the region into squares or triangles and feed these
C     small elements to drawpoly()

        do 200 j=1,nny-1
           jj = j
           do 100 i=1,nnx-1
              ii = i

              IF (interrupted) RETURN
     
C     do upper triangle
C     

C     For curvilinear modulo axes:
C     Replicate the first point in a column/row if necessary
              if (i.eq.nx) ii = 1
              if (j.eq.ny) jj = 1

              zt(1)=z(ii,jj)
              xt(1) = x(i)
              yt(1) = y(j)

              if (sqfflg) then
     
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
                 zt(3)=z(ii+1,jj)
                 xt(3)=xt(2)
                 yt(3)=yt(1)
     
                 call drawpoly(xt,yt,zt,zlev,nlev,3)

              endif

 100        continue
 200     continue

C     
C     Generate a color/gray scale key.
C     
        if (do_key .AND. .NOT.interrupted) call key ! 1/89 - *sh*

        return
        end


C======================================================================

c:    
c:    Output triangle for contour display.
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
#else
        INCLUDE         'tmap_pplv11inc:tmap_miss.inc'
#endif


      zmin=  1e30
      zmax= -1e30

      do 10 i = 1,nsides
	if ((zle .and. z(i) .le. cmzle)  .or.
     .      (zeq .and. z(i) .eq. cmzeq)  .or.
     .      (zge .and. z(i) .ge. cmzge)) return 

         zmin=amin1(zmin,z(i))
         zmax=amax1(zmax,z(i))
 10   continue

c 1.96 CODE PREVIOUS TO FIX
c      zmin=amin1(z(1),z(2))
c      zmax=amax1(z(1),z(2))
c      do 10 i = 3,nsides
c         zmin=amin1(zmin,z(i))
c         zmax=amax1(zmax,z(i))
c 10   continue
c
c     if(zmax.ge.0.9e35)return
c
c 1.96 END OF PRE-FIX CODE

      do 100 i = 1, nlev-1

         if ((zmin.gt.zlev(i+1)) .or. (zmax.lt.zlev(i))) goto 100

         if ((zmin.ge.zlev(i)) .and. (zmax.le.zlev(i+1))) then
            do 30 j = 1, nsides
               xpoly(j)=x(j)
               ypoly(j)=y(j)
 30         continue
            call fillpoly(x, y, nsides, i-1)
            return
         endif
c
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

      subroutine sqrfill(z,x,y,nx,ny,zlev,nlev)
      integer nx,ny,nlev
      real z(nx,ny),x(nx),y(ny),zlev(nlev)
c
      real xt(4),yt(4)
      integer i,j,index
c
      do 10 i=1,nx
         do 20 j=1,ny
c
c build square
c
            if(i.eq.1)then
               xt(1)=(x(1)-(x(2)-x(1))*0.5)
               xt(2)=(x(1)+x(2))*0.5
            else if(i.eq.nx) then
               xt(1)=(x(nx-1)+x(nx))*0.5
               xt(2)=(x(nx)+(x(nx)-x(nx-1))*0.5)
            else
               xt(1)=(x(i-1)+x(i))*0.5
               xt(2)=(x(i)+x(i+1))*0.5
            endif
            xt(3)=xt(2)
            xt(4)=xt(1)
c
            if(j.eq.1)then
               yt(1)=(y(1)-(y(2)-y(1))*0.5)
               yt(3)=(y(1)+y(2))*0.5
            else if(j.eq.ny) then
               yt(1)=(y(ny-1)+y(ny))*0.5
               yt(3)=(y(ny)+(y(ny)-y(ny-1))*0.5)
            else
               yt(1)=(y(j-1)+y(j))*0.5
               yt(3)=(y(j)+y(j+1))*0.5
            endif
            yt(2)=yt(1)
            yt(4)=yt(3)
c
c find index
c
            if(z(i,j).lt.zlev(1)) then
               index=1
               goto 40
            else if(z(i,j).gt.zlev(nlev))then
               index=nlev
               goto 40
            endif
            do 30 index=1,nlev-1
c
               if(z(i,j).ge.zlev(index) .and.
     *              z(i,j).lt.zlev(index+1)) goto 40
 30         continue
c
 40         call fillpoly(xt,yt,4,index-1)
 20      continue
 10   continue
      return
      end
