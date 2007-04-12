	subroutine conseg(z,nxs,nys,px,py,nx,ny,xf,xoff,yf,yoff,
     *	zlev,ndecl,lwgtl,lclr,nlev,hgt,narc,ang,status) 
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
C**    @(#)conseg.F	1.2    7/30/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c
C 
C       lots of places ---> PPLMOD_PPLCMD ---> PPLMOD_PLTIT 
C       ---> PPLMOD_PLOTZ ---> PPLMOD_CONSEG
C       
C       This subroutine plots contours of grid z(i,j) in square segments
C       of length lseg grid units per side.
C       
C       z      - actual 2-D array of data values to be contoured
C       nxs    - number of X grid points
C       nys    - number of Y grid points
C       px     - array of grid points along the X axis
C       py     - array of grid points along the Y axis
C       nx     - number of X grid points
C       ny     - number of Y grid points
C       xf     - factor to go from X axis to page coordinates?
C       xoff   - X axis offset in page coordinates?
C       yf     - factor to go from Y axis to page coordinates?
C       yoff   - Y axis offset in page coordinates?
C       zlev   - array of contour level values (lo val --> hi val)
C       ndecl  - number of decimal places in the contour labels
C       lwgtl  - weight of the contour line
C       lclr   - 
C       nlev   - number of levels to use
C       hgt    - height of characters of labels in inches
C       narc   - number of arc subsegments to use in contouring
C       ang    - angle to rotate the contour plot
C       
C       
C	oceanography emr  dec/1969 
C       *jc* 5.97 Major clean up and commenting
C       *jc* 5.97 Added code for curvilinear coordinates
C      *acm* v581 5/2005 Add CONTOUR/MOD capability for modulo
C                 replications on curvilinear data
C V602 *acm* 2/07 check status coming out of CURV_COORD_RANGE; part of fix to bug 1485
C V602 *acm* 4/07 ferr_ok needs to be declared (in errmsg.parm)

	include 'MISS.INC'
        include 'xcurvilinear.cmn'
        include 'errmsg.parm'

	dimension z(nxs,nys),px(nx),py(ny)
	dimension zlev(2),ndecl(2),lwgtl(2),lclr(2)

        real xadd, lmin, lmax, big
	integer error,trans_no, ilo, ihi, jlo, jhi, status
	real xinv,yinv,w(4),v(4),ndc(4),uc(4)

C       Calculate center point about which to rotate plot
C       Plot will be rotated by 'ang' degrees.

	xc = (px(1)+px(nx))*0.5
	yc = (py(1)+py(ny))*0.5

* Get range of indices for the curvi X,Y coordinates actually needed for the 
* hlimits, vlimits that were requested.

*	GET PROPER SCALING, NDC COORDINATES, ETC FOR FILL
        IF (curvilinear) THEN 
	   CALL gqcntn (error,trans_no)
	   IF (error .NE. 0) GOTO 1000

	   CALL gqnt (trans_no,error,w,v)
	   IF (error .NE. 0) GOTO 1000

	   CALL get_user_coords (w,v,ndc,uc,xinv,yinv)

           CALL CURV_COORD_RANGE (uc, ilo, ihi, jlo, jhi, status )
           IF (status .NE. ferr_ok) goto 1000
        ENDIF
      
        big = ABS(bad_val4)
        lmin = big
        lmax = -1*big
        xadd = 0.

        CALL CONSEG_SECTION(xc, yc, nx, ny, nxs, nys, px, py, z, 
     .    xf, xoff, yf, yoff, zlev, ndecl, lwgtl, lclr, nlev, hgt, 
     .    narc, ang, ilo, ihi, jlo, jhi, xadd, lmin, lmax)

        IF (xmodulo .AND. curvilinear) THEN
           nleft = 0
           nright = 0
           IF (lmin .GT. uc(1) ) nleft =  1 + INT(lmin - uc(1) )/ 360
           IF (lmax .LT. uc(2) ) nright = 1 + INT(uc(2) - lmax )/ 360

           IF (nleft .GT. 0) THEN
              xadd = 0.
              DO 100 i = 1, nleft
                 xadd = xadd - 360.
                 CALL CONSEG_SECTION(xc, yc, nx, ny, nxs, nys, px, py,z,
     .          xf, xoff, yf, yoff, zlev, ndecl, lwgtl, lclr, nlev, hgt, 
     .          narc, ang, ilo, ihi, jlo, jhi, xadd, lmin, lmax)

 100          CONTINUE
           ENDIF

           IF (nright .GT. 0) THEN
              xadd = 0.
              DO 110 i = 1, nright
                 xadd = xadd + 360.
                 CALL CONSEG_SECTION(xc, yc, nx, ny, nxs, nys, px, py,z,
     .          xf, xoff, yf, yoff, zlev, ndecl, lwgtl, lclr, nlev, hgt, 
     .          narc, ang, ilo, ihi, jlo, jhi, xadd, lmin, lmax)
 110          CONTINUE
           ENDIF


        ENDIF   ! xmodulo



1000    return 
	end 
