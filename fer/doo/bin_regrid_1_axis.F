	SUBROUTINE BIN_REGRID_1_AXIS(cx_lims, axis, trans,
     .				 src, msrc, dst, mdst, ss21, src_cx,
     .                           dst_cx, wksize )
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
* 	grid dst_grid by binning withing boxes along one axis
*       @BIN returns binned data, @NBIN returns number of valid source
*       points in the destination grid cell

* 	msrc and mdst point to data structures in COMMON/XVARIABLES/ which

* 	specify the subscript bounds of src and dst within their 
* 	respective grids.

*	ss21 holds the indices of source grid boxes (on src_grid) for box
*	limits on dst_grid.

* 	programmer - Ansley Manke
* 	NOAA/PMEL,Seattle,WA - TMAP/ SDIG  11/2/2015
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xcontext.cmn'
	include 'xmem_subsc.cmn'
	include 'xvariables.cmn'

* calling argument declarations:
	INTEGER cx_lims, axis, trans, msrc, mdst, wksize
	INTEGER ss21(wksize)

	REAL    src( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .	             m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .	        dst( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .	             m2lot:m2hit,m2loe:m2hie,m2lof:m2hif )

* local variable declarations:

        LOGICAL         ok_neg111, coord_in_dst
        INTEGER		dstss,srcss,src_grid,dst_grid,i,j,k,l,m,n
	INTEGER	     	srcss_lo,srcss_hi, dstss_lo, dstss_hi,
     .                  src_cx, dst_cx

	REAL		ddist, bad_src, bad_dst
	REAL*8		TM_WORLD, TDEST_WORLD, sum,
     .			xsrc_min, xsrc_max, 
     .			xdst_lo, xdst_hi, xsrc_pt, nsrc_in_dst
******************************************************************************

* initialize
	src_grid = mr_grid( msrc )
	dst_grid = mr_grid( mdst )

* limits for calculation
        srcss_lo = mr_lo_ss(msrc,axis)
        srcss_hi = mr_hi_ss(msrc,axis)

* Range of subscripts might be negative, and contain the value of unspecified_int4 
* among them.  ok_neg111 distinguishes between this and the flag unspecified_int4.

        ok_neg111 = srcss_lo .LT. unspecified_int4 .AND.
     .              srcss_hi .GT. unspecified_int4

        dstss_lo = cx_lo_ss(cx_lims,axis)
        dstss_hi = cx_hi_ss(cx_lims,axis)

* flag for bad/missing data
        bad_src = mr_bad_data(msrc)
        bad_dst = mr_bad_data(mdst)

* determine the source grid boxes containing the destination box edges
	CALL GET_AVE_LIMS( srcss_lo, srcss_hi, src_grid,
     .			   dstss_lo, dstss_hi, dst_grid,
     .			   axis, ss21 )

* along X axis
        IF ( axis .EQ. x_dim ) THEN
           DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
           DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
           DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)

* loop through the each destination grid box
	   DO 160 dstss = dstss_lo,dstss_hi
	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

*  check if requested region is out of source range
              IF ( (srcss_lo .EQ. unspecified_int4
     .	     .OR.  srcss_hi .EQ. unspecified_int4)
     .       .AND. .NOT.(ok_neg111) ) THEN
	         dst(dstss,j,k,l,m,n) = bad_dst
	         goto 160
	      ENDIF

	      xdst_lo = TM_WORLD(dstss, dst_grid, axis, box_lo_lim )
	      xdst_hi = TM_WORLD(dstss, dst_grid, axis, box_hi_lim )

* add all partial or complete boxes of data from the source grid

* IF source box completely encloses the dest box, then srcss_lo=srcss_hi
* and we can just stick the source point into the destination (for avg or var)
	      sum = 0.0D0
	      nsrc_in_dst = 0.0D0
	      DO 110 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(srcss, j, k, l, m, n) .EQ. bad_src ) GOTO 110

	         xsrc_pt = TM_WORLD(srcss,src_grid,axis,box_middle)
	         coord_in_dst = (xsrc_pt.GE.xdst_lo .AND. xsrc_pt.LT.xdst_hi) 

* ... source point in dest box
                 IF (coord_in_dst) THEN
	            nsrc_in_dst = nsrc_in_dst + 1
	            sum   = sum  + src(srcss, j, k, l, m, n)
	         ENDIF 
 
 110	      CONTINUE


* ... divide by npts in bin for @BIN or return # pts for @NBIN

	      IF (trans.EQ.prgrd_bin ) THEN

	         IF ( nsrc_in_dst .EQ. 0.0 ) THEN
	            dst(dstss, j, k, l, m, n) = bad_dst
	        ELSE
	            dst(dstss, j, k, l, m, n) = sum / nsrc_in_dst
	        ENDIF

	      ELSEIF ( trans.EQ.prgrd_nbin ) THEN
	         dst(dstss, j, k, l, m, n) = nsrc_in_dst
	      ENDIF

 160	   CONTINUE
 	   ENDDO  ! Close loops over all dimensions
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

* along Y axis
        ELSEIF ( axis .EQ. Y_dim ) THEN

           DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
           DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
           DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* loop through the each destination grid box
	   DO 260 dstss = dstss_lo,dstss_hi
	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

*  check if requested region is out of source range
              IF ( (srcss_lo .EQ. unspecified_int4
     .	     .OR.  srcss_hi .EQ. unspecified_int4)
     .       .AND. .NOT.(ok_neg111) ) THEN
	         dst(i,dstss,k,l,m,n) = bad_dst
	         goto 260
	      ENDIF

	      xdst_lo  = TM_WORLD(dstss, dst_grid, axis, box_lo_lim )
	      xdst_hi  = TM_WORLD(dstss, dst_grid, axis, box_hi_lim )

* add all partial or complete boxes of data from the source grid

* IF source box completely encloses the dest box, then srcss_lo=srcss_hi
* and we can just stick the source point into the destination (for avg or var)
	      sum = 0.0D0
	      nsrc_in_dst = 0.0D0
	      DO 210 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, srcss, k, l, m, n) .EQ. bad_src ) GOTO 210

	         xsrc_pt = TM_WORLD(srcss,src_grid,axis,box_middle)
	         coord_in_dst = (xsrc_pt.GE.xdst_lo .AND. xsrc_pt.LT.xdst_hi)

* ... source point in dest box
                 IF (coord_in_dst) THEN
	            nsrc_in_dst = nsrc_in_dst + 1
	            sum   = sum  + src(i, srcss, k, l, m, n)
	         ENDIF 
 
 210	      CONTINUE

* ... divide by npts in bin for @BIN or return # pts for @NBIN

	      IF (trans.EQ.prgrd_bin ) THEN
	         IF ( nsrc_in_dst .EQ. 0.0 ) THEN
	            dst(i, dstss, k, l, m, n) = bad_dst
	         ELSE
	            dst(i, dstss, k, l, m, n) = sum / nsrc_in_dst
	         ENDIF
	      ELSEIF ( trans.EQ.prgrd_nbin ) THEN
	         dst(i, dstss, k, l, m, n) = nsrc_in_dst
	      ENDIF

 260	   CONTINUE
 	   ENDDO  ! Close loops over all dimensions
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

* along Z axis
        ELSEIF ( axis .EQ. z_dim ) THEN

           DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
           DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
           DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
           DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* loop through the each destination grid box
	   DO 360 dstss = dstss_lo,dstss_hi
	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

*  check if requested region is out of source range
              IF ( (srcss_lo .EQ. unspecified_int4
     .	     .OR.  srcss_hi .EQ. unspecified_int4)
     .       .AND. .NOT.(ok_neg111) ) THEN
	         dst(i,j,dstss,k,m,n) = bad_dst
	         goto 360
	      ENDIF

	      xdst_lo = TM_WORLD(dstss, dst_grid, axis, box_lo_lim )
	      xdst_hi = TM_WORLD(dstss, dst_grid, axis, box_hi_lim )

* add all partial or complete boxes of data from the source grid

* IF source box completely encloses the dest box, then srcss_lo=srcss_hi
* and we can just stick the source point into the destination (for avg or var)
	      sum = 0.0D0
	      nsrc_in_dst = 0.0D0
	      DO 310 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, j, srcss, l, m, n) .EQ. bad_src ) GOTO 310

	         xsrc_pt = TM_WORLD(srcss,src_grid,axis,box_middle)

	         coord_in_dst = (xsrc_pt.GE.xdst_lo .AND. xsrc_pt.LT.xdst_hi)

* ... source point in dest box
                 IF (coord_in_dst) THEN
	            nsrc_in_dst = nsrc_in_dst + 1
	            sum   = sum  + src(i, j, srcss, l, m, n)
	         ENDIF 
 
 310	      CONTINUE

* ... divide by npts in bin for @BIN or return # pts for @NBIN

	      IF (trans.EQ.prgrd_bin ) THEN
	         IF ( nsrc_in_dst .EQ. 0.0 ) THEN
	            dst(i, j, dstss, l, m, n) = bad_dst
	         ELSE
	            dst(i, j, dstss, l, m, n) = sum / nsrc_in_dst
	         ENDIF
	      ELSEIF ( trans.EQ.prgrd_nbin ) THEN
	         dst(i, j, dstss, l, m, n) = nsrc_in_dst
	      ENDIF

 360	   CONTINUE
 	   ENDDO  ! Close loops over all dimensions
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

* along T axis
        ELSEIF ( axis .EQ. T_dim ) THEN

           DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
           DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
           DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
           DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* loop through the each destination grid box
	   DO 460 dstss = dstss_lo,dstss_hi
	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

*  check if requested region is out of source range
              IF ( (srcss_lo .EQ. unspecified_int4
     .	     .OR.  srcss_hi .EQ. unspecified_int4)
     .       .AND. .NOT.(ok_neg111) ) THEN
	         dst(i,j,k,dstss,m,n) = bad_dst
	         goto 460
	      ENDIF

* USE SOURCE GRID TIME/DATE ENCODINGS FOR ALL CALCULATIONS 6/94
* initialization of TDEST_WORLD is done from GET_AVE_LIMS

	      xdst_lo = TDEST_WORLD(dstss, dst_grid, axis, box_lo_lim)
	      xdst_hi = TDEST_WORLD(dstss, dst_grid, axis, box_hi_lim)

* add all partial or complete boxes of data from the source grid

* IF source box completely encloses the dest box, then srcss_lo=srcss_hi
* and we can just stick the source point into the destination (for avg or var)
	      sum = 0.0D0
	      nsrc_in_dst = 0.0
	      DO 410 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, j, k, srcss, m, n) .EQ. bad_src ) GOTO 410
	         xsrc_pt = TM_WORLD(srcss,src_grid,axis,box_middle)

	        coord_in_dst = (xsrc_pt.GE.xdst_lo .AND. xsrc_pt.LT.xdst_hi) 

* ... source point in dest box
                 IF (coord_in_dst) THEN
	            nsrc_in_dst = nsrc_in_dst + 1
	            sum   = sum  + src(i, j, k, srcss, m, n)
	         ENDIF 

 410	      CONTINUE

* ... divide by npts in bin for @BIN or return # pts for @NBIN

	      IF (trans.EQ.prgrd_bin ) THEN
	         IF ( nsrc_in_dst .EQ. 0.0 ) THEN
	            dst(i, j, k, dstss, m, n) = bad_dst
	         ELSE
	            dst(i, j, k, dstss, m, n) = sum / nsrc_in_dst
	         ENDIF
	      ELSEIF ( trans.EQ.prgrd_nbin ) THEN
	         dst(i, j, k, dstss, m, n) = nsrc_in_dst
	      ENDIF

 460	   CONTINUE
 	   ENDDO  ! Close loops over all dimensions
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

* along E axis
        ELSEIF ( axis .EQ. e_dim ) THEN

           DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
           DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
           DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* loop through the each destination grid box
	   DO 560 dstss = dstss_lo,dstss_hi
	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

*  check if requested region is out of source range
              IF ( (srcss_lo .EQ. unspecified_int4
     .	     .OR.  srcss_hi .EQ. unspecified_int4)
     .       .AND. .NOT.(ok_neg111) ) THEN
	         dst(i,j,k,l,dstss,n) = bad_dst
	         goto 560
	      ENDIF

* USE SOURCE GRID TIME/DATE ENCODINGS FOR ALL CALCULATIONS 6/94
* initialization of TDEST_WORLD is done from GET_AVE_LIMS

	      xdst_lo = TDEST_WORLD(dstss, dst_grid, axis, box_lo_lim)
	      xdst_hi = TDEST_WORLD(dstss, dst_grid, axis, box_hi_lim)

* add all partial or complete boxes of data from the source grid

* IF source box completely encloses the dest box, then srcss_lo=srcss_hi
* and we can just stick the source point into the destination (for avg or var)
	      sum = 0.0D0
	      nsrc_in_dst = 0.0
	      DO 510 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, j, k, l, srcss, n) .EQ. bad_src ) GOTO 510

	         xsrc_pt = TM_WORLD(srcss,src_grid,axis,box_middle)
	         coord_in_dst = (xsrc_pt.GE.xdst_lo .AND. xsrc_pt.LT.xdst_hi) 

* ... source point in dest box
                 IF (coord_in_dst) THEN
	            nsrc_in_dst = nsrc_in_dst + 1
	            sum   = sum  + src(i, j, k, l, srcss, n)
	         ENDIF 
 510	      CONTINUE

* ... divide by npts in bin for @BIN or return # pts for @NBIN

	      IF (trans.EQ.prgrd_bin ) THEN
	         IF ( nsrc_in_dst .EQ. 0.0 ) THEN
	            dst(i, j, k, l, dstss, n) = bad_dst
	         ELSE
	            dst(i, j, k, l, dstss, n) = sum / nsrc_in_dst
	         ENDIF
	      ELSEIF ( trans.EQ.prgrd_nbin ) THEN
	         dst(i, j, k, l, dstss, n) = nsrc_in_dst
	      ENDIF

 560	   CONTINUE
 	   ENDDO  ! Close loops over all dimensions
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

* along F axis
        ELSEIF ( axis .EQ. f_dim ) THEN

           DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
           DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
           DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* loop through the each destination grid box
	   DO 660 dstss = dstss_lo,dstss_hi
	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

*  check if requested region is out of source range
              IF ( (srcss_lo .EQ. unspecified_int4
     .	     .OR.  srcss_hi .EQ. unspecified_int4)
     .       .AND. .NOT.(ok_neg111) ) THEN
	         dst(i,j,k,l,m,dstss) = bad_dst
	         goto 660
	      ENDIF

* USE SOURCE GRID TIME/DATE ENCODINGS FOR ALL CALCULATIONS 6/94
* initialization of TDEST_WORLD is done from GET_AVE_LIMS

	      xdst_lo = TDEST_WORLD(dstss, dst_grid, axis, box_lo_lim)
	      xdst_hi = TDEST_WORLD(dstss, dst_grid, axis, box_hi_lim)

* add all partial or complete boxes of data from the source grid

* IF source box completely encloses the dest box, then srcss_lo=srcss_hi
* and we can just stick the source point into the destination (for avg or var)
	      sum = 0.0D0
	      nsrc_in_dst = 0.0
	      DO 610 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, j, k, l, m, srcss) .EQ. bad_src ) GOTO 610

	         xsrc_pt = TM_WORLD(srcss,src_grid,axis,box_middle)
	         coord_in_dst = (xsrc_pt.GE.xdst_lo .AND. xsrc_pt.LT.xdst_hi) 

* ... source point in dest box
                 IF (coord_in_dst) THEN
	            nsrc_in_dst = nsrc_in_dst + 1
	            sum   = sum  + src(i, j, k, l, m, srcss)
	         ENDIF 
 610	      CONTINUE

* ... divide by npts in bin for @BIN or return # pts for @NBIN

	      IF (trans.EQ.prgrd_bin ) THEN
	         IF ( nsrc_in_dst .EQ. 0.0 ) THEN
	            dst(i, j, k, l, m, dstss) = bad_dst
	         ELSE
	            dst(i, j, k, l, m, dstss) = sum / nsrc_in_dst
	         ENDIF
	      ELSEIF ( trans.EQ.prgrd_nbin ) THEN
	         dst(i, j, k, l, m, dstss) = nsrc_in_dst
	      ENDIF

 660	   CONTINUE
 	   ENDDO  ! Close loops over all dimensions
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

        ENDIF

	RETURN
	END
