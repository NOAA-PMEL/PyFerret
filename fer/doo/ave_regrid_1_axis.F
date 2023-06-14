	SUBROUTINE AVE_REGRID_1_AXIS( cx_lims, axis, trans,
     .				 dst_lox, dst_hix, dst_loy, dst_hiy,
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
* 	grid dst_grid by averaging withing boxes along one axis

* 	msrc and mdst point to data structures in COMMON/XVARIABLES/ which

* 	specify the subscript bounds of src and dst within their 
* 	respective grids.

*	ss21 holds the indices of source grid boxes (on src_grid) for box
*	limits on dst_grid.

* 	programmer - Steve Hankin
* 	NOAA/PMEL,Seattle,WA - Tropical Modeling and Analysis Program

* note: the conversion to 4D is of the Q&D variety.  Optimization is possible
*	by providing additional work space and pre-computing the box sizes
*	along the X and Y axes

* V312:	 6/7/94 - use TDST_WORLD for simpler, faster date conversions
*	speed other axes, too, by not using AXIS_DISTANCE (units divide out)
* V420: 10/95 - handles regridding by @VAR, as well
*	 1/96 - handles @SUM and @NGD as well
* V510: *sh* 1/00 variance of a grid box with only a single value
*	contributing is undefined
* v541: *acm* 3/02 In some cases the source indices include a range
*                  outside the destination indices, so ddist < 0.  Test
*                  for this and only have a contribution to the avg from 
*                  portions of the source axis that overlap the dest box.
* v552  *acm* 4/03 Use src_lo_ss and src_hi_ss for the range on the src
*                  axis, as a more complete fix of the above bug.
* v570 *acm* 5/04 Range of subscripts might be negative, and contain 
*                 -111 among them. Allow regridding, using the flag 
*                 ok_neg111 to distinguish this case from the flag 
*                 unspecified_int4.
* v580 *acm* 11/04 Fix the definition of ok_neg111; if -111 is at
*                  either end of the interval we cannot distinguish
*                  it as a valid index from its use as the bad flag.
* NOTE: -111 in the above was the value of unspecified_int4 (now -999)
*      *kms*  2/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* v685 *acm* 06/13 Fix ticket 2070: regridding when source cell encloses 
*                  dest cell.
* v685 *acm*  7/13 Fix bug in F axis loop, and fix nonX directions in the
*                  code for source cell enclosing dest cell.
* V741  5/18 *acm* issue 1868: gridding transforms that will grid to same axis
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xcontext.cmn'
	include 'xmem_subsc.cmn'
	include 'xvariables.cmn'

* calling argument declarations:
	INTEGER cx_lims, axis, trans, msrc, mdst,
     .          dst_lox, dst_hix, dst_loy, dst_hiy , wksize
	INTEGER ss21(wksize)

	REAL    src( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .	             m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .	        dst( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .	             m2lot:m2hit,m2loe:m2hie,m2lof:m2hif )

* local variable declarations:

        LOGICAL         IS_SELF_REGRID, ok_neg111, src_encloses, self_vari
        INTEGER		dstss,srcss,src_grid,dst_grid,i,j,k,l,m,n
	INTEGER	     	srcss_lo,srcss_hi, dstss_lo,dstss_hi, ncontrib,
     .                  src_cx, dst_cx

	REAL		ddist, bad_src, bad_dst
	REAL*8		TM_WORLD, TDEST_WORLD, sum, distance, dev,
     .			xsrc_min, xsrc_max, xsrc_lo, xsrc_hi,
     .			xdst_lo, xdst_hi
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

* Is this a regridding to the same line: Compute a result if @VAR on same axis,
* 0 for valid data, missing otherwise.

	self_vari = IS_SELF_REGRID (axis, src_grid, dst_grid, trans )

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
	      distance = 0.0D0
	      DO 110 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(srcss, j, k, l, m, n) .EQ. bad_src ) GOTO 110

	         xsrc_min = TM_WORLD(srcss,src_grid,axis,box_lo_lim)
	         xsrc_max = TM_WORLD(srcss,src_grid,axis,box_hi_lim)

	        src_encloses = 
     .               (xdst_lo.GT.xsrc_min .AND. xdst_hi.LT.xsrc_max) 

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi )

	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box
                 IF (ddist .GT. 0.) THEN
	            distance  = distance + ddist
	            IF (trans .EQ. prgrd_sum ) THEN
	               sum   = sum  + src(srcss, j, k, l, m, n)
     .			         * ddist/(xsrc_max-xsrc_min)
	            ELSEIF (trans .EQ. prgrd_good_pt ) THEN
	               sum   = sum  + ddist/(xsrc_max-xsrc_min)
	            ELSE	! average or variance
	               sum   = sum  + ddist * src(srcss, j, k, l, m, n)
	            ENDIF
                 ENDIF

 110	      CONTINUE

	      IF ( trans.EQ.prgrd_good_pt ) THEN
	         dst(dstss, j, k, l, m, n) = sum
	      ELSEIF ( distance .EQ. 0.0 ) THEN
	         dst(dstss, j, k, l, m, n) = bad_dst
	      ELSEIF ( trans.EQ.prgrd_sum ) THEN
	         dst(dstss, j, k, l, m, n) = sum
	      ELSE
* ... divide by distance to get average
	         dst(dstss, j, k, l, m, n) = sum / distance
		 IF (src_encloses) dst(dstss,j,k,l,m,n) = 
     .			         src(srcss_lo,j,k,l,m,n)
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
	      distance = 0.0D0
	      DO 210 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, srcss, k, l, m, n) .EQ. bad_src ) GOTO 210

	         xsrc_min = TM_WORLD(srcss,src_grid,axis,box_lo_lim)
	         xsrc_max = TM_WORLD(srcss,src_grid,axis,box_hi_lim)

	        src_encloses = 
     .               (xdst_lo.GT.xsrc_min .AND. xdst_hi.LT.xsrc_max) 

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi)

	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box
                 IF (ddist .GT. 0.) THEN
	            distance  = distance + ddist
	            IF (trans .EQ. prgrd_sum ) THEN
	               sum   = sum  + src(i, srcss, k, l, m, n)
     .			         * ddist/(xsrc_max-xsrc_min)
	            ELSEIF (trans .EQ. prgrd_good_pt ) THEN
	               sum   = sum  + ddist/(xsrc_max-xsrc_min)
	            ELSE	! average or variance
	               sum   = sum  + ddist * src(i, srcss, k, l, m, n)
	            ENDIF
                 ENDIF

 210	      CONTINUE

	      IF (trans.EQ.prgrd_good_pt ) THEN
	         dst(i, dstss, k, l, m, n) = sum
	      ELSEIF ( distance .EQ. 0.0 ) THEN
	         dst(i, dstss, k, l, m, n) = bad_dst
	      ELSEIF (trans.EQ.prgrd_sum ) THEN
	         dst(i, dstss, k, l, m, n) = sum
	      ELSE
* ... divide by distance to get average
	         dst(i, dstss, k, l, m, n) = sum / distance
		 IF (src_encloses) dst(i,dstss,k,l,m,n) = 
     .			         src(i,srcss_lo,k,l,m,n)
	      ENDIF

 260	   CONTINUE
	   ENDDO
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
	      distance = 0.0D0
	      DO 310 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, j, srcss, l, m, n) .EQ. bad_src ) GOTO 310

	         xsrc_min = TM_WORLD(srcss,src_grid,axis,box_lo_lim)
	         xsrc_max = TM_WORLD(srcss,src_grid,axis,box_hi_lim)

	        src_encloses = 
     .               (xdst_lo.GT.xsrc_min .AND. xdst_hi.LT.xsrc_max) 

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi )

	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box
                 IF (ddist .GT. 0.) THEN
	            distance  = distance + ddist
	            IF (trans .EQ. prgrd_sum ) THEN
	               sum   = sum  + src(i, j, srcss, l, m, n)
     .			         * ddist/(xsrc_max-xsrc_min)
	            ELSEIF (trans .EQ. prgrd_good_pt ) THEN
	               sum   = sum  + ddist/(xsrc_max-xsrc_min)
	            ELSE	! average or variance
	               sum   = sum  + ddist * src(i, j, srcss, l, m, n)
	            ENDIF
                 ENDIF

 310	      CONTINUE

	      IF ( trans.EQ.prgrd_good_pt ) THEN
	         dst(i, j, dstss, l, m, n) = sum
	      ELSEIF ( distance .EQ. 0.0 ) THEN
	         dst(i, j, dstss, l, m, n) = bad_dst
	      ELSEIF (trans.EQ.prgrd_sum ) THEN
	         dst(i, j, dstss, l, m, n) = sum
	      ELSE
* ... divide by distance to get average
	         dst(i, j, dstss, l, m, n) = sum / distance
		 IF (src_encloses) dst(i,j,dstss,l,m,n) = 
     .			         src(i,j,srcss_lo,l,m,n)
	      ENDIF

 360	   CONTINUE
	   ENDDO
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
	      distance = 0.0D0
	      DO 410 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, j, k, srcss, m, n) .EQ. bad_src ) GOTO 410

	         xsrc_min = TM_WORLD(srcss,src_grid,axis,box_lo_lim)
	         xsrc_max = TM_WORLD(srcss,src_grid,axis,box_hi_lim)

	        src_encloses = 
     .               (xdst_lo.GT.xsrc_min .AND. xdst_hi.LT.xsrc_max) 

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi)

	         ddist   = xsrc_hi - xsrc_lo

* ... contribution from this source box
                 IF (ddist .GT. 0.) THEN
	            distance  = distance + ddist
	            IF (trans .EQ. prgrd_sum ) THEN
	               sum   = sum  + src(i, j, k, srcss, m, n)
     .			         * ddist/(xsrc_max-xsrc_min)
	            ELSEIF (trans .EQ. prgrd_good_pt ) THEN
	               sum   = sum  + ddist/(xsrc_max-xsrc_min)
	            ELSE	! average or variance
	               sum   = sum  + ddist * src(i, j, k, srcss, m, n)
	            ENDIF
	         ENDIF 

 410	      CONTINUE

	      IF ( trans.EQ.prgrd_good_pt ) THEN
	         dst(i, j, k, dstss, m, n) = sum
	      ELSEIF ( distance .EQ. 0.0 ) THEN
	         dst(i, j, k, dstss, m, n) = bad_dst
	      ELSEIF (trans.EQ.prgrd_sum ) THEN
	         dst(i, j, k, dstss, m, n) = sum
	      ELSE
* ... divide by distance to get average
	         dst(i, j, k, dstss, m, n) = sum / distance
		 IF (src_encloses) dst(i,j,k,dstss,m,n) = 
     .			         src(i,j,k,srcss_lo,m,n)
	      ENDIF

 460	   CONTINUE
	   ENDDO
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
	      distance = 0.0D0
	      DO 510 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, j, k, l, srcss, n) .EQ. bad_src ) GOTO 510

	         xsrc_min = TM_WORLD(srcss,src_grid,axis,box_lo_lim)
	         xsrc_max = TM_WORLD(srcss,src_grid,axis,box_hi_lim)

	        src_encloses = 
     .               (xdst_lo.GT.xsrc_min .AND. xdst_hi.LT.xsrc_max) 

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi)

	         ddist   = xsrc_hi - xsrc_lo

* ... contribution from this source box
                 IF (ddist .GT. 0.) THEN
	            distance  = distance + ddist
	            IF (trans .EQ. prgrd_sum ) THEN
	               sum   = sum  + src(i, j, k, l, srcss, n)
     .			         * ddist/(xsrc_max-xsrc_min)
	            ELSEIF (trans .EQ. prgrd_good_pt ) THEN
	               sum   = sum  + ddist/(xsrc_max-xsrc_min)
	            ELSE	! average or variance
	               sum   = sum  + ddist * src(i, j, k, l, srcss, n)
	            ENDIF
	         ENDIF
 510	      CONTINUE

	      IF ( trans.EQ.prgrd_good_pt ) THEN
	         dst(i, j, k, l, dstss, n) = sum
	      ELSEIF ( distance .EQ. 0.0 ) THEN
	         dst(i, j, k, l, dstss, n) = bad_dst
	      ELSEIF (trans.EQ.prgrd_sum ) THEN
	         dst(i, j, k, l, dstss, n) = sum
	      ELSE
* ... divide by distance to get average
	         dst(i, j, k, l, dstss, n) = sum / distance
		 IF (src_encloses) dst(i,j,k,l,dstss,n) = 
     .			         src(i,j,k,l,srcss_lo,n)
	      ENDIF

 560	   CONTINUE
	   ENDDO
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
	      distance = 0.0D0
	      DO 610 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, j, k, l, m, srcss) .EQ. bad_src ) GOTO 610

	         xsrc_min = TM_WORLD(srcss,src_grid,axis,box_lo_lim)
	         xsrc_max = TM_WORLD(srcss,src_grid,axis,box_hi_lim)

	        src_encloses = 
     .               (xdst_lo.GT.xsrc_min .AND. xdst_hi.LT.xsrc_max) 

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi)

	         ddist   = xsrc_hi - xsrc_lo

* ... contribution from this source box
                 IF (ddist .GT. 0.) THEN
	            distance  = distance + ddist
	            IF (trans .EQ. prgrd_sum ) THEN
	               sum   = sum  + src(i, j, k, l, m, srcss)
     .			         * ddist/(xsrc_max-xsrc_min)
	            ELSEIF (trans .EQ. prgrd_good_pt ) THEN
	               sum   = sum  + ddist/(xsrc_max-xsrc_min)
	            ELSE	! average or variance
	               sum   = sum  + ddist * src(i, j, k, l, m, srcss)
	            ENDIF
	         ENDIF
 610	      CONTINUE

	      IF ( trans.EQ.prgrd_good_pt ) THEN
	         dst(i, j, k, l, m, dstss) = sum
	      ELSEIF ( distance .EQ. 0.0 ) THEN
	         dst(i, j, k, l, m, dstss) = bad_dst
	      ELSEIF (trans.EQ.prgrd_sum ) THEN
	         dst(i, j, k, l, m, dstss) = sum
	      ELSE
* ... divide by distance to get average
	         dst(i, j, k, l, m, dstss) = sum / distance
		 IF (src_encloses) dst(i,j,k,l,m,dstss) = 
     .			         src(i,j,k,l,m,srcss_lo)
	      ENDIF

 660	   CONTINUE
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* TODO: along E and F axes?

        ENDIF

	IF (trans .NE. prgrd_variance) RETURN

**************************************************************
**************************************************************
**************************************************************
* pass 2:  determine variance
**************************************************************
**************************************************************
**************************************************************
	
* along X axis
        IF ( axis .EQ. x_dim ) THEN
           DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
           DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
           DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)

* loop through the each destination grid box
	   DO 1160 dstss = dstss_lo,dstss_hi

	      IF ( dst(dstss, j, k, l, m, n) .EQ. bad_dst ) GOTO 1160

	      IF (self_vari) THEN
	         dst(dstss, j, k, l, m, n) = 0.0
		 GOTO 1160
	      ENDIF

	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

	      xdst_lo = TM_WORLD(dstss, dst_grid, axis, box_lo_lim)
	      xdst_hi = TM_WORLD(dstss, dst_grid, axis, box_hi_lim)

* add all partial or complete boxes of data from the source grid
	      sum = 0.0D0
	      distance = 0.0D0
	      ncontrib = 0
	      DO 1110 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(srcss, j, k, l, m, n) .EQ. bad_src ) GOTO 1110

	         xsrc_lo = MAX( TM_WORLD(srcss,src_grid,axis,box_lo_lim),
     .                          xdst_lo )
	         xsrc_hi = MIN( TM_WORLD(srcss,src_grid,axis,box_hi_lim),
     .                          xdst_hi )
	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box
 	         IF (ddist .GT. 0.0) THEN
	           ncontrib = ncontrib + 1
	           distance  = distance + ddist
	           dev = DBLE(src(srcss, j, k, l, m, n)) -
     .                   DBLE(dst(dstss, j, k, l, m, n))
	           sum = sum  + ddist * dev*dev
	         ENDIF

 1110	      CONTINUE

* ... divide by weight to get variance
	      IF (ncontrib .GT. 1) THEN
	         dst(dstss, j, k, l, m, n) = sum / distance
	      ELSE
	         dst(dstss, j, k, l, m, n) = bad_dst
	      ENDIF

 1160	   CONTINUE
	   ENDDO
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
	   DO 1260 dstss = dstss_lo,dstss_hi

	      IF ( dst(i, dstss, k, l, m, n) .EQ. bad_dst ) GOTO 1260

	      IF (self_vari) THEN
	         dst(i, dstss, k, l, m, n) = 0.0
		 GOTO 1260
	      ENDIF

	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

	      xdst_lo = TM_WORLD(dstss, dst_grid, axis, box_lo_lim)
	      xdst_hi = TM_WORLD(dstss, dst_grid, axis, box_hi_lim)

* add all partial or complete boxes of data from the source grid
	      sum = 0.0D0
	      distance = 0.0D0
	      ncontrib = 0
	      DO 1210 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, srcss, k, l, m, n) .EQ. bad_src ) GOTO 1210

	         xsrc_lo = MAX( TM_WORLD(srcss,src_grid,axis,box_lo_lim),
     .                          xdst_lo )
	         xsrc_hi = MIN( TM_WORLD(srcss,src_grid,axis,box_hi_lim),
     .                          xdst_hi )
	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box
	         IF (ddist .GT. 0.0) THEN 
	           ncontrib = ncontrib + 1
	           distance  = distance + ddist
	           dev = DBLE(src(i, srcss, k, l, m, n)) -
     .                   DBLE(dst(i, dstss, k, l, m, n))
	           sum   = sum  + ddist * dev*dev
	         ENDIF

 1210	      CONTINUE

* ... divide by weight to get variance
	      IF (ncontrib .GT. 1) THEN
	         dst(i, dstss, k, l, m, n) = sum / distance
	      ELSE
	         dst(i, dstss, k, l, m, n) = bad_dst
	      ENDIF

 1260	   CONTINUE
	   ENDDO
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
	   DO 1360 dstss = dstss_lo,dstss_hi

	      IF ( dst(i, j, dstss, l, m, n) .EQ. bad_dst ) GOTO 1360

	      IF (self_vari) THEN
	         dst(i, j, dstss, l, m, n) = 0.0
		 GOTO 1360
	      ENDIF

	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

	      xdst_lo = TM_WORLD(dstss, dst_grid, axis, box_lo_lim)
	      xdst_hi = TM_WORLD(dstss, dst_grid, axis, box_hi_lim)

* add all partial or complete boxes of data from the source grid
	      sum = 0.0D0
	      distance = 0.0D0
	      ncontrib = 0
	      DO 1310 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, j, srcss, l, m, n) .EQ. bad_src ) GOTO 1310

	         xsrc_lo = MAX( TM_WORLD(srcss,src_grid,axis,box_lo_lim),
     .                          xdst_lo )
	         xsrc_hi = MIN( TM_WORLD(srcss,src_grid,axis,box_hi_lim),
     .                          xdst_hi )
	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box
 	         IF (ddist .GT. 0.0) THEN
	           ncontrib = ncontrib + 1
	           distance  = distance + ddist
	           dev = DBLE(src(i, j, srcss, l, m, n)) -
     .                   DBLE(dst(i, j, dstss, l, m, n))
	           sum = sum  + ddist * dev*dev
	         ENDIF

 1310	      CONTINUE

* ... divide by weight to get variance
	      IF (ncontrib .GT. 1) THEN
	         dst(i, j, dstss, l, m, n) = sum / distance
	      ELSE
	         dst(i, j, dstss, l, m, n) = bad_dst
	      ENDIF

 1360	   CONTINUE
	   ENDDO
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
	   DO 1460 dstss = dstss_lo,dstss_hi

	      IF ( dst(i, j, k, dstss, m, n) .EQ. bad_dst ) GOTO 1460

	      IF (self_vari) THEN
	         dst(i, j, k, dstss, m, n) = 0.0
		 GOTO 1460
	      ENDIF

	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

* USE SOURCE GRID TIME/DATE ENCODINGS FOR ALL CALCULATIONS 6/94
* initialization of TDEST_WORLD is done from GET_AVE_LIMS

	      xdst_lo = TDEST_WORLD(dstss, dst_grid, axis, box_lo_lim)
	      xdst_hi = TDEST_WORLD(dstss, dst_grid, axis, box_hi_lim)

* add all partial or complete boxes of data from the source grid
	      sum = 0.0D0
	      distance = 0.0D0
	      ncontrib = 0
	      DO 1410 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, j, k, srcss, m, n) .EQ. bad_src ) GOTO 1410

	         xsrc_lo = MAX( TM_WORLD(srcss,src_grid,axis,box_lo_lim),
     .                          xdst_lo )
	         xsrc_hi = MIN( TM_WORLD(srcss,src_grid,axis,box_hi_lim),
     .                          xdst_hi )
	         ddist   = xsrc_hi - xsrc_lo
                 IF (ddist .LT. 0.) ddist = 0.

* ... contribution from this source box
 	         IF (ddist .GT. 0.0) THEN
	           ncontrib = ncontrib + 1
	           distance  = distance + ddist
	           dev = DBLE(src(i, j, k, srcss, m, n)) -
     .                   DBLE(dst(i, j, k, dstss, m, n))
	           sum = sum  + ddist * dev*dev
	         ENDIF

 1410	      CONTINUE

* ... divide by weight to get variance
	      IF (ncontrib .GT. 1) THEN
	         dst(i, j, k, dstss, m, n) = sum / distance
	      ELSE
	         dst(i, j, k, dstss, m, n) = bad_dst
	      ENDIF

 1460	   CONTINUE
	   ENDDO
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
	   DO 1560 dstss = dstss_lo,dstss_hi

	      IF ( dst(i, j, k, l, dstss, n) .EQ. bad_dst ) GOTO 1560

	      IF (self_vari) THEN
	         dst(i, j, k, l, dstss, n) = 0.0
		 GOTO 1560
	      ENDIF

	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

* USE SOURCE GRID TIME/DATE ENCODINGS FOR ALL CALCULATIONS 6/94
* initialization of TDEST_WORLD is done from GET_AVE_LIMS

	      xdst_lo = TDEST_WORLD(dstss, dst_grid, axis, box_lo_lim)
	      xdst_hi = TDEST_WORLD(dstss, dst_grid, axis, box_hi_lim)

* add all partial or complete boxes of data from the source grid
	      sum = 0.0D0
	      distance = 0.0D0
	      ncontrib = 0
	      DO 1510 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, j, k, l, srcss, n) .EQ. bad_src ) GOTO 1510

	         xsrc_lo = MAX( TM_WORLD(srcss,src_grid,axis,box_lo_lim),
     .                          xdst_lo )
	         xsrc_hi = MIN( TM_WORLD(srcss,src_grid,axis,box_hi_lim),
     .                          xdst_hi )
	         ddist   = xsrc_hi - xsrc_lo
                 IF (ddist .LT. 0.) ddist = 0.

* ... contribution from this source box
 	         IF (ddist .GT. 0.0) THEN
	           ncontrib = ncontrib + 1
	           distance  = distance + ddist
	           dev = DBLE(src(i, j, k, l, srcss, n)) -
     .                   DBLE(dst(i, j, k, l, dstss, n))
	           sum = sum  + ddist * dev*dev
	         ENDIF

 1510	      CONTINUE

* ... divide by weight to get variance
	      IF (ncontrib .GT. 1) THEN
	         dst(i, j, k, l, dstss, n) = sum / distance
	      ELSE
	         dst(i, j, k, l, dstss, n) = bad_dst
	      ENDIF

 1560	   CONTINUE
	   ENDDO
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
	   DO 1660 dstss = dstss_lo,dstss_hi

	      IF ( dst(i, j, k, l, m, dstss) .EQ. bad_dst ) GOTO 1660

	      IF (self_vari) THEN
	         dst(i, j, k, l, m, dstss) = 0.0
		 GOTO 1660
	      ENDIF

	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

* USE SOURCE GRID TIME/DATE ENCODINGS FOR ALL CALCULATIONS 6/94
* initialization of TDEST_WORLD is done from GET_AVE_LIMS

	      xdst_lo = TDEST_WORLD(dstss, dst_grid, axis, box_lo_lim)
	      xdst_hi = TDEST_WORLD(dstss, dst_grid, axis, box_hi_lim)

* add all partial or complete boxes of data from the source grid
	      sum = 0.0D0
	      distance = 0.0D0
	      ncontrib = 0
	      DO 1610 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, j, k, l, m, srcss) .EQ. bad_src ) GOTO 1610

	         xsrc_lo = MAX( TM_WORLD(srcss,src_grid,axis,box_lo_lim),
     .                          xdst_lo )
	         xsrc_hi = MIN( TM_WORLD(srcss,src_grid,axis,box_hi_lim),
     .                          xdst_hi )
	         ddist   = xsrc_hi - xsrc_lo
                 IF (ddist .LT. 0.) ddist = 0.

* ... contribution from this source box
 	         IF (ddist .GT. 0.0) THEN
	           ncontrib = ncontrib + 1
	           distance  = distance + ddist
	           dev = DBLE(src(i, j, k, l, m, srcss)) -
     .                   DBLE(dst(i, j, k, l, m, dstss))
	           sum = sum  + ddist * dev*dev
	         ENDIF

 1610	      CONTINUE

* ... divide by weight to get variance
	      IF (ncontrib .GT. 1) THEN
	         dst(i, j, k, l, m, dstss) = sum / distance
	      ELSE
	         dst(i, j, k, l, m, dstss) = bad_dst
	      ENDIF

 1660	   CONTINUE
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
        ENDIF

	RETURN
	END
