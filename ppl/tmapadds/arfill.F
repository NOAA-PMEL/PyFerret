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
*     v5.81 *acm* 4/05 Changes for FILL/MOD. Draw replications for curvilinear datasets

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
        INCLUDE         'errmsg.parm'

        integer nx,ny
        real z(nx,ny),x(nx),y(ny)

        real x_extra,y_extra
        integer i,nnx,nny, ilo, ihi, jlo, jhi, status
	real xadd,xinv,yinv,w(4),v(4),ndc(4),uc(4)

        real lmin, lmax

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

C acm 4/05 When plotting curvilinear data with FILL/MODULO put this logic into effect

        IF (xmodulo) dim1_modulo = .TRUE.
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

* Get range of indices for the curvi X,Y coordinates actually needed for the 
* hlimits, vlimits that were requested.


*	GET PROPER SCALING, NDC COORDINATES, ETC FOR FILL
	CALL gqcntn (error,trans_no)
	IF (error .NE. 0) GOTO 1000

	CALL gqnt (trans_no,error,w,v)
	IF (error .NE. 0) GOTO 1000

	CALL get_user_coords (w,v,ndc,uc,xinv,yinv)

        IF (curvilinear) 
     .     CALL CURV_COORD_RANGE (uc, ilo, ihi, jlo, jhi, status )
      
        big = ABS(bad_val4)
        lmin = big
        lmax = -1*big
        xadd = 0.

       CALL ARFILL_SECTION(x, y, z, zlev, nx, ny, nlev, nnx, nny, 
     .       x_extra, y_extra, lmin, lmax, ilo, ihi, jlo, jhi, xadd, 
     .       status)

       if (status .eq. ferr_interrupt) go to 1000

        IF (xmodulo .AND. curvilinear) THEN
           nleft = 0
           nright = 0
           IF (lmin .GT. uc(1) ) nleft =  1 + INT(lmin - uc(1) )/ 360
           IF (lmax .LT. uc(2) ) nright = 1 + INT(uc(2) - lmax )/ 360

           IF (nleft .GT. 0) THEN
              xadd = 0.
              DO 100 i = 1, nleft
                 xadd = xadd - 360.
                 CALL ARFILL_SECTION(x, y, z, zlev, nx, ny, nlev, 
     .               nnx, nny, x_extra, y_extra, lmin, lmax, ilo, 
     .               ihi, jlo, jhi, xadd, status)
                 IF (status .EQ. ferr_interrupt) GOTO 1000
 100          CONTINUE
           ENDIF

           IF (nright .GT. 0) THEN
              xadd = 0.
              DO 110 i = 1, nright
                 xadd = xadd + 360.
                 CALL ARFILL_SECTION(x, y, z, zlev, nx, ny, nlev, 
     .               nnx, nny, x_extra, y_extra, lmin, lmax, ilo, 
     .               ihi, jlo, jhi, xadd, status)
                 IF (status .EQ. ferr_interrupt) GOTO 1000
 110          CONTINUE
           ENDIF
         
        ENDIF   ! xmodulo
C     
C     Generate a color/gray scale key.
C     Restore standard key style afterward (do this here, to 
C     catch both contour and contour/set calls).
C     
        if (do_key .AND. .NOT.interrupted) THEN
            call key ! 1/89 - *sh*
           CALL SHADE_KEY_CONT (0)
        ENDIF

1000    return
        end
