      subroutine arfill(z,x,y,nx,ny)
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
*     V5.81 ACM 3/05  Other routines (drawpoly, buffer_fillpoly, flush_buffered_fillpoly)
*                     removed to their own files.

	include 	'ARFILL.INC'
        include         'GKSCM1.INC'
        include         'gkscm2.cmn'
        include         'miss_inc.decl'
        include         'MISS.INC'
        include         'xcurvilinear.cmn'

        include         'xinterrupt.cmn' ! 1/89 *sh*
        include         'ppl_in_ferret.cmn'
        include         'shade_vars.cmn'

	include		'parampl5_dat.decl'
	include		'PARAMPL5.DAT'
	include		'cont_inc.decl'
	include		'CONT.INC'

        integer nx,ny
        real z(nx,ny),x(nx),y(ny)

        real xt(5),yt(5),zt(5),x_extra,y_extra
        integer i,j,ii,jj,nnx,nny


C     *jd* mod to allow filling single level fields
        if (nlev .le. 1) return

	shd_levels = nlev - 1
	shd_levels = MAX(nlev2-1,1) ! this is how it is in SHADE.F

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

                 call drawpoly(xt,yt,zt,4)

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
     
                 call drawpoly(xt,yt,zt,3)
c     
c     do lower triangle
c     
                 upperflg = .false.

                 zt(3)=z(ii+1,jj)
                 xt(3)=xt(2)
                 yt(3)=yt(1)
     
                 call drawpoly(xt,yt,zt,3)

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
