	SUBROUTINE IIN_REGRID_1_AXIS( cx_lims, idim, src, msrc, 
     .				 dst, mdst, ss21, src_cx, dst_cx, 
     .				 wksize, status )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* 	Using data values from src organized on grid src_grid, regrid to
* 	grid dst_grid by integrating within boxes along one axis
*
* 	msrc and mdst point to data structures in COMMON/XVARIABLES/ which
*
* 	specify the subscript bounds of src and dst within their 
* 	respective grids.
*
*	ss21 holds the indices of source grid boxes (on src_grid) for box
*	limits on dst_grid.
*
*	Return results in world-coordinate units

* 	programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Science Data Integration Group
* revision 0.0 - 5/2018  from ave_regrid_1_axis
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xcontext.cmn'
	include 'xmem_subsc.cmn'
	include 'xvariables.cmn'
	include 'errmsg.parm'
	include 'xunits.cmn_text'
	include 'xtm_grid.cmn_text'

* calling argument declarations:
	INTEGER cx_lims, idim, msrc, mdst, wksize,
     .                  src_cx, dst_cx, status
	INTEGER ss21(wksize)

	REAL    src( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .	             m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .	        dst( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .	             m2lot:m2hit,m2loe:m2hie,m2lof:m2hif )

* local variable declarations:

        LOGICAL         IS_SELF_REGRID, GEOG_COS_FACTOR, 
     .			ok_neg111, src_encloses, need_cos, bad, lmod
        INTEGER		dstss,srcss,src_grid,dst_grid,i,j,k,l,m,n
	INTEGER	     	srcss_lo,srcss_hi, dstss_lo,dstss_hi, ncontrib, 
     .			usrc, udst

	REAL		TM_WORLD, TDEST_WORLD, sum,
     .			xsrc_min, xsrc_max, xsrc_lo, xsrc_hi,
     .			xdst_lo, xdst_hi, ddist, bad_src, bad_dst,
     .			unit_src, unit_dst, bx, ybxlo, ybxhi, cos_factor
******************************************************************************

* initialize
	src_grid = mr_grid( msrc )
	dst_grid = mr_grid( mdst )

* limits for calculation
        srcss_lo = mr_lo_ss(msrc,idim)
        srcss_hi = mr_hi_ss(msrc,idim)

* Handle the axis as non-modulo, so that we start the integration correctly
* and not reaching into the next grid cell below the start.

	lmod = line_modulo( grid_line(idim,src_grid) ) 
	IF (lmod) line_modulo( grid_line(idim,src_grid) ) = .FALSE.

* Range of subscripts might be negative, and contain the value of unspecified_int4 
* among them.  ok_neg111 distinguishes between this and the flag unspecified_int4.

        ok_neg111 = srcss_lo .LT. unspecified_int4 .AND.
     .              srcss_hi .GT. unspecified_int4

        dstss_lo = cx_lo_ss(cx_lims,idim)
        dstss_hi = cx_hi_ss(cx_lims,idim)

	
* perform integrations on standard units if possible
	usrc = line_unit_code(grid_line(idim,src_grid))
	udst = line_unit_code(grid_line(idim,dst_grid))
	unit_src = un_convert( usrc )
	unit_dst = un_convert( udst )

* Check: Same or compatible units?
        bad = usrc* udst .LT. 0  ! one is time and one is not

* Can convert time, but not converting e.g. between km and mile
	IF (.NOT.bad .AND. udst.GE.0)  bad = (unit_src .NE. unit_dst) 

	IF (bad) GOTO 5100

* flag for bad/missing data
        bad_src = mr_bad_data(msrc)
        bad_dst = mr_bad_data(mdst)

* determine the source grid boxes containing the destination box edges
	CALL GET_IIN_LIMS( srcss_lo, srcss_hi, src_grid,
     .			   dstss_lo, dstss_hi, dst_grid,
     .			   idim, ss21 )

* along X axis
        IF ( idim .EQ. x_dim ) THEN

           cos_factor = 1.
           need_cos = GEOG_COS_FACTOR( idim, src_grid )
     .                .AND. cx_lo_s2(cx_lims).NE.unspecified_int4

           DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
           DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
           DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)

           IF ( need_cos ) THEN
              ybxlo = deg2rad* 
     .              TM_WORLD(j,src_grid,y_dim,box_lo_lim)
              ybxhi = deg2rad* 
     .              TM_WORLD(j,src_grid,y_dim,box_hi_lim)
              bx = ABS(ybxhi - ybxlo)
              cos_factor = ABS( SIN(ybxhi) - SIN(ybxlo) )/ bx
           ENDIF 

* loop through the each destination grid box
	   sum = 0.0D0
	   dst(dstss_lo, j, k, l, m, n) = sum
	   DO dstss = dstss_lo,dstss_hi
	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

*  check if requested region is out of source range
              IF ( (srcss_lo .EQ. unspecified_int4
     .	     .OR.  srcss_hi .EQ. unspecified_int4)
     .       .AND. .NOT.(ok_neg111) ) THEN
	         dst(dstss,j,k,l,m,n) = bad_dst
	         CYCLE
	      ENDIF

* Get destination edges to use

	      IF (dstss .EQ. dstss_lo) THEN
 	         xdst_lo = unit_dst* TM_WORLD(dstss, dst_grid, idim, box_lo_lim )
		 xdst_hi = unit_dst* TM_WORLD(dstss, dst_grid, idim, box_middle )
	      ELSE
	         xdst_lo =  unit_dst* TM_WORLD(dstss-1, dst_grid, idim, box_middle )
	         xdst_hi = unit_dst* TM_WORLD(dstss, dst_grid, idim, box_middle )
	      ENDIF

* add all partial or complete boxes of data from the source grid

* IF source box completely encloses the dest box, then srcss_lo=srcss_hi
* and we can just add the source point into the sum
	      DO srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(srcss, j, k, l, m, n) .EQ. bad_src ) CYCLE

	         xsrc_min = unit_src* TM_WORLD(srcss,src_grid,idim,box_lo_lim)
	         xsrc_max = unit_src* TM_WORLD(srcss,src_grid,idim,box_hi_lim)

	        src_encloses = 
     .               (xdst_lo.GT.xsrc_min .AND. xdst_hi.LT.xsrc_max) 

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi )

	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box
                 IF (ddist .GT. 0.) THEN
	            sum   = sum  + cos_factor* ddist * src(srcss, j, k, l, m, n)
                 ENDIF

	      ENDDO
              dst(dstss, j, k, l, m, n) = sum

	   ENDDO  ! dstss_lo,dstss_hi

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* along Y axis
        ELSEIF ( idim .EQ. Y_dim ) THEN

           DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
           DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
           DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* loop through the each destination grid box
	   sum = 0.0D0
           dst(i, dstss_lo, k, l, m, n) = sum
	   DO dstss = dstss_lo,dstss_hi
	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

*  check if requested region is out of source range
              IF ( (srcss_lo .EQ. unspecified_int4
     .	     .OR.  srcss_hi .EQ. unspecified_int4)
     .       .AND. .NOT.(ok_neg111) ) THEN
	         dst(i,dstss,k,l,m,n) = bad_dst
	         CYCLE
	      ENDIF

* Get destination edges to use

	      IF (dstss .EQ. dstss_lo) THEN
 	         xdst_lo = unit_dst* TM_WORLD(dstss, dst_grid, idim, box_lo_lim )
		 xdst_hi = unit_dst* TM_WORLD(dstss, dst_grid, idim, box_middle )
	      ELSE
	         xdst_lo = unit_dst* TM_WORLD(dstss-1, dst_grid, idim, box_middle )
	         xdst_hi = unit_dst* TM_WORLD(dstss, dst_grid, idim, box_middle )
	      ENDIF

* add all partial or complete boxes of data from the source grid

* IF source box completely encloses the dest box, then srcss_lo=srcss_hi
* and we can just add the source point into the sum
	      DO srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, srcss, k, l, m, n) .EQ. bad_src ) CYCLE

	         xsrc_min = unit_src* TM_WORLD(srcss,src_grid,idim,box_lo_lim)
	         xsrc_max = unit_src* TM_WORLD(srcss,src_grid,idim,box_hi_lim)

	        src_encloses = 
     .               (xdst_lo.GT.xsrc_min .AND. xdst_hi.LT.xsrc_max) 

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi)

	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box
                 IF (ddist .GT. 0.) THEN
	            sum = sum + ddist*  src(i, srcss, k, l, m, n)
                 ENDIF

              ENDDO
              dst(i, dstss, k, l, m, n) = sum

	   ENDDO  ! dstss_lo,dstss_hi

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* along Z axis
        ELSEIF ( idim .EQ. z_dim ) THEN

           DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
           DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
           DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
           DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* loop through the each destination grid box
	   sum = 0.0D0
           dst(i, j, dstss_lo, l, m, n) = sum
	   DO dstss = dstss_lo,dstss_hi
	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

*  check if requested region is out of source range
              IF ( (srcss_lo .EQ. unspecified_int4
     .	     .OR.  srcss_hi .EQ. unspecified_int4)
     .       .AND. .NOT.(ok_neg111) ) THEN
	         dst(i,j,dstss,k,m,n) = bad_dst
	         CYCLE
	      ENDIF

* Get destination edges to use

	      IF (dstss .EQ. dstss_lo) THEN
 	         xdst_lo = unit_dst* TM_WORLD(dstss, dst_grid, idim, box_lo_lim )
		 xdst_hi = unit_dst* TM_WORLD(dstss, dst_grid, idim, box_middle )
	      ELSE
	         xdst_lo = unit_dst* TM_WORLD(dstss-1, dst_grid, idim, box_middle )
	         xdst_hi = unit_dst* TM_WORLD(dstss, dst_grid, idim, box_middle )
	      ENDIF

* add all partial or complete boxes of data from the source grid

* IF source box completely encloses the dest box, then srcss_lo=srcss_hi
* and we can just add the source point into the sum
	      DO srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, j, srcss, l, m, n) .EQ. bad_src ) CYCLE

	         xsrc_min = unit_src* TM_WORLD(srcss,src_grid,idim,box_lo_lim)
	         xsrc_max = unit_src* TM_WORLD(srcss,src_grid,idim,box_hi_lim)

	        src_encloses = 
     .               (xdst_lo.GT.xsrc_min .AND. xdst_hi.LT.xsrc_max) 

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi )

	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box
                 IF (ddist .GT. 0.) THEN
	            sum = sum + ddist*  src(i, j, srcss, l, m, n)
                 ENDIF

              ENDDO
	      dst(i, j, dstss, l, m, n) = sum

	   ENDDO  ! dstss_lo,dstss_hi

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* along T axis
        ELSEIF ( idim .EQ. T_dim ) THEN

           DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
           DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
           DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
           DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* loop through the each destination grid box
	   sum = 0.0D0
           dst(i, j, k, dstss_lo, m, n) = sum
	   DO dstss = dstss_lo,dstss_hi
	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

*  check if requested region is out of source range
              IF ( (srcss_lo .EQ. unspecified_int4
     .	     .OR.  srcss_hi .EQ. unspecified_int4)
     .       .AND. .NOT.(ok_neg111) ) THEN
	         dst(i,j,k,dstss,m,n) = bad_dst
	         CYCLE
	      ENDIF

* Get destination edges to use

* Get destination edges to use

	      IF (dstss .EQ. dstss_lo) THEN
 	         xdst_lo = unit_dst* TM_WORLD(dstss, dst_grid, idim, box_lo_lim )
		 xdst_hi = unit_dst* TM_WORLD(dstss, dst_grid, idim, box_middle )
	      ELSE
	         xdst_lo = unit_dst* TM_WORLD(dstss-1, dst_grid, idim, box_middle )
	         xdst_hi = unit_dst* TM_WORLD(dstss, dst_grid, idim, box_middle )
	      ENDIF

* add all partial or complete boxes of data from the source grid

* IF source box completely encloses the dest box, then srcss_lo=srcss_hi
* and we can just add the source point into the sum
	      DO srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, j, k, srcss, m, n) .EQ. bad_src ) CYCLE

	         xsrc_min = unit_src* TM_WORLD(srcss,src_grid,idim,box_lo_lim)
	         xsrc_max = unit_src* TM_WORLD(srcss,src_grid,idim,box_hi_lim)

	        src_encloses = 
     .               (xdst_lo.GT.xsrc_min .AND. xdst_hi.LT.xsrc_max) 

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi)

	         ddist   = xsrc_hi - xsrc_lo

* ... contribution from this source box
                 IF (ddist .GT. 0.) THEN
	            sum = sum + ddist*  src(i, j, k, srcss, m, n)
	         ENDIF 

              ENDDO
	      dst(i, j, k, dstss, m, n) = sum

	   ENDDO  ! dstss_lo,dstss_hi

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* along E axis
        ELSEIF ( idim .EQ. e_dim ) THEN

           DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
           DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
           DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* loop through the each destination grid box
	   sum = 0.0D0
           dst(i, j, k, l, dstss_lo, n) = sum
	   DO dstss = dstss_lo,dstss_hi
	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

*  check if requested region is out of source range
              IF ( (srcss_lo .EQ. unspecified_int4
     .	     .OR.  srcss_hi .EQ. unspecified_int4)
     .       .AND. .NOT.(ok_neg111) ) THEN
	         dst(i,j,k,l,dstss,n) = bad_dst
	         CYCLE
	      ENDIF

* Get destination edges to use

* Get destination edges to use

	      IF (dstss .EQ. dstss_lo) THEN
 	         xdst_lo = unit_dst* TM_WORLD(dstss, dst_grid, idim, box_lo_lim )
		 xdst_hi = unit_dst* TM_WORLD(dstss, dst_grid, idim, box_middle )
	      ELSE
	         xdst_lo = unit_dst* TM_WORLD(dstss-1, dst_grid, idim, box_middle )
	         xdst_hi = unit_dst* TM_WORLD(dstss, dst_grid, idim, box_middle )
	      ENDIF

* add all partial or complete boxes of data from the source grid

* IF source box completely encloses the dest box, then srcss_lo=srcss_hi
* and we can just add the source point into the sum
	      DO srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, j, k, l, srcss, n) .EQ. bad_src ) CYCLE

	         xsrc_min = unit_src* TM_WORLD(srcss,src_grid,idim,box_lo_lim)
	         xsrc_max = unit_src* TM_WORLD(srcss,src_grid,idim,box_hi_lim)

	        src_encloses = 
     .               (xdst_lo.GT.xsrc_min .AND. xdst_hi.LT.xsrc_max) 

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi)

	         ddist   = xsrc_hi - xsrc_lo

* ... contribution from this source box
                 IF (ddist .GT. 0.) THEN
	            sum = sum + ddist*  src(i, j, k, l, srcss, n)
	         ENDIF
              ENDDO
	      dst(i, j, k, l, dstss, n) = sum

	   ENDDO  ! dstss_lo,dstss_hi

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* along F axis
        ELSEIF ( idim .EQ. f_dim ) THEN

           DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
           DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
           DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* loop through the each destination grid box
	   sum = 0.0D0
           dst(i, j, k, l, m, dstss_lo) = sum
	   DO dstss = dstss_lo,dstss_hi
	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

*  check if requested region is out of source range
              IF ( (srcss_lo .EQ. unspecified_int4
     .	     .OR.  srcss_hi .EQ. unspecified_int4)
     .       .AND. .NOT.(ok_neg111) ) THEN
	         dst(i,j,k,l,m,dstss) = bad_dst
	         CYCLE
	      ENDIF

* Get destination edges to use

* Get destination edges to use

	      IF (dstss .EQ. dstss_lo) THEN
 	         xdst_lo = unit_dst* TM_WORLD(dstss, dst_grid, idim, box_lo_lim )
		 xdst_hi = unit_dst* TM_WORLD(dstss, dst_grid, idim, box_middle )
	      ELSE
	         xdst_lo = unit_dst* TM_WORLD(dstss-1, dst_grid, idim, box_middle )
	         xdst_hi = unit_dst* TM_WORLD(dstss, dst_grid, idim, box_middle )
	      ENDIF

* add all partial or complete boxes of data from the source grid

* IF source box completely encloses the dest box, then srcss_lo=srcss_hi
* and we can just add the source point into the sum
	      DO srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, j, k, l, m, srcss) .EQ. bad_src ) CYCLE

	         xsrc_min = unit_src* TM_WORLD(srcss,src_grid,idim,box_lo_lim)
	         xsrc_max = unit_src* TM_WORLD(srcss,src_grid,idim,box_hi_lim)

	        src_encloses = 
     .               (xdst_lo.GT.xsrc_min .AND. xdst_hi.LT.xsrc_max) 

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi)

	         ddist   = xsrc_hi - xsrc_lo

* ... contribution from this source box
                 IF (ddist .GT. 0.) THEN
	            sum = sum + ddist*  src(i, j, k, l, m, srcss)
	         ENDIF
              ENDDO
	      dst(i, j, k, l, m, dstss) = sum

	   ENDDO  ! dstss_lo,dstss_hi

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

        ENDIF

 5000	CONTINUE
	line_modulo( grid_line(idim,src_grid) ) = lmod
	RETURN

 5100   status = ferr_invalid_command
	GOTO 5000
	END
