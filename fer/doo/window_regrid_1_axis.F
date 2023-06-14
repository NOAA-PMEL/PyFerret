	SUBROUTINE WINDOW_REGRID_1_AXIS( cx_lims, axis, trans,
     .				 dst_lox, dst_hix, dst_loy, dst_hiy,
     .				 src, msrc, dst, mdst, sslo, sshi,
     .				 wksize )

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
* 	grid dst_grid within boxes along one axis

*	The algorithm used for regridding is determined by "trans", the
*	regridding transformation -- e.g. min, max, ...
*	This routine handles only algorithms that compute using discrete
*	(unweighted) points.  Transforms such as average and variance are
*	handled in ave_regrid_1_axis

* 	msrc and mdst point to data structures in COMMON/XVARIABLES/ which
* 	specify the subscript bounds of src and dst within their 
* 	respective grids.

*	ss21 holds the indices of source grid boxes (on src_grid) for box
*	limits on dst_grid.

* 	programmer - Steve Hankin
* 	NOAA/PMEL,Seattle,WA - Tropical Modeling and Analysis Program

* V420: 1/96 - removed SUM and NGD from here to ave_regrid_1_axis (weighted)
* V500: 4/99 *kob* bug fix:  fixed loops for y,z,t axis to loop
*                            only to srcss_hi-1
* V510: 1/00 *sh* bug fix to behavior at edges of region
* v570 *acm* 5/04 Note: we do not need to use the ok_neg111 fix for 
*                 when the range of subscripts might be negative, and 
*                 contain -111 among them. Tests for unspecified_int4
*                 are not used in this routine.
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* V685+ 12/13 *acm* Typo in the F-axis code.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xcontext.cmn'
	include	'xmem_subsc.cmn'
	include 'xvariables.cmn'


* calling argument declarations:
	INTEGER	cx_lims, axis, msrc, trans, mdst,
     .          dst_lox, dst_hix, dst_loy, dst_hiy , wksize
	INTEGER	sslo(wksize), sshi(wksize)

	REAL    src( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          dst( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif )

* local variable declarations:
	INTEGER	npts, dstss, srcss, src_grid, dst_grid,
     .          i, j, k, l, m, n
	INTEGER	srcss_lo,srcss_hi, dstss_lo,dstss_hi,
     .		src_lo, src_hi

	REAL	accum, bad_src, bad_dst

******************************************************************************

* initialize
	src_grid = mr_grid( msrc )
	dst_grid = mr_grid( mdst )

* limits for calculation
        srcss_lo = mr_lo_ss(msrc,axis)
        srcss_hi = mr_hi_ss(msrc,axis)

        dstss_lo = cx_lo_ss(cx_lims,axis)
        dstss_hi = cx_hi_ss(cx_lims,axis)

* flag for bad/missing data
        bad_src = mr_bad_data(msrc)
        bad_dst = mr_bad_data(mdst)

* determine the source grid boxes containing the destination boxes
	CALL GET_WINDOW_LIMS ( srcss_lo, srcss_hi, src_grid,
     .			   dstss_lo, dstss_hi, dst_grid,
     .			   axis, sslo, sshi )


* along X axis
        IF ( axis .EQ. x_dim ) THEN
           DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
           DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
           DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)

* loop through the each destination grid box
	      DO dstss = dstss_lo,dstss_hi

	         src_lo = sslo(dstss-dstss_lo+1)
	         src_hi = sshi(dstss-dstss_lo+1)

	         IF     ( trans .EQ. prgrd_min ) THEN
	            accum = arbitrary_large_val4
	         ELSEIF ( trans .EQ. prgrd_max ) THEN
	            accum = arbitrary_small_val4
	         ENDIF

	         npts = 0
	         DO srcss = src_lo, src_hi

* ... ignore missing source data
	            IF ( src(srcss,j,k,l,m,n) .EQ. bad_src ) CYCLE

* ... contribution from this source box
	            IF     ( trans .EQ. prgrd_min ) THEN
	               accum = MIN( accum, src(srcss,j,k,l,m,n) )
	            ELSEIF ( trans .EQ. prgrd_max ) THEN
	               accum = MAX( accum, src(srcss,j,k,l,m,n) )
	           ENDIF
	           npts = npts + 1
	         ENDDO

                 IF ( npts .GT. 0 ) THEN
                    dst(dstss,j,k,l,m,n) = accum
                 ELSE
                 dst(dstss,j,k,l,m,n) = bad_dst
                 ENDIF

	      ENDDO

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
	      DO dstss = dstss_lo,dstss_hi

	         src_lo = sslo(dstss-dstss_lo+1)
	         src_hi = sshi(dstss-dstss_lo+1)

	         IF     ( trans .EQ. prgrd_min ) THEN
	            accum = arbitrary_large_val4
	         ELSEIF ( trans .EQ. prgrd_max ) THEN
	            accum = arbitrary_small_val4
	         ENDIF
	         npts = 0

	         DO srcss = src_lo, src_hi

* ... ignore missing source data
	            IF ( src(i,srcss,k,l,m,n) .EQ. bad_src ) CYCLE

* ... contribution from this source box
	            IF     ( trans .EQ. prgrd_min ) THEN
	               accum = MIN( accum, src(i,srcss,k,l,m,n) )
	            ELSEIF ( trans .EQ. prgrd_max ) THEN
	               accum = MAX( accum, src(i,srcss,k,l,m,n) )
	            ENDIF
	            npts = npts + 1
	         ENDDO

                 IF ( npts .GT. 0 ) THEN
                    dst(i,dstss,k,l,m,n) = accum
                 ELSE
                    dst(i,dstss,k,l,m,n) = bad_dst
                 ENDIF

	      ENDDO

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
	      DO dstss = dstss_lo,dstss_hi

	         src_lo = sslo(dstss-dstss_lo+1)
	         src_hi = sshi(dstss-dstss_lo+1)

	         IF     ( trans .EQ. prgrd_min ) THEN
	            accum = arbitrary_large_val4
	         ELSEIF ( trans .EQ. prgrd_max ) THEN
	            accum = arbitrary_small_val4
	         ENDIF
	         npts = 0

	         DO srcss = src_lo, src_hi

* ... ignore missing source data
	            IF ( src(i,j,srcss,l,m,n) .EQ. bad_src ) CYCLE

* ... contribution from this source box
	            IF     ( trans .EQ. prgrd_min ) THEN
	               accum = MIN( accum, src(i,j,srcss,l,m,n) )
	            ELSEIF ( trans .EQ. prgrd_max ) THEN
	               accum = MAX( accum, src(i,j,srcss,l,m,n) )
	            ENDIF
	            npts = npts + 1
                 ENDDO

                 IF ( npts .GT. 0 ) THEN
                    dst(i,j,dstss,l,m,n) = accum
                 ELSE
                    dst(i,j,dstss,l,m,n) = bad_dst
                 ENDIF

	      ENDDO

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
	      DO dstss = dstss_lo,dstss_hi

	         src_lo = sslo(dstss-dstss_lo+1)
	         src_hi = sshi(dstss-dstss_lo+1)

	         IF     ( trans .EQ. prgrd_min ) THEN
	            accum = arbitrary_large_val4
	         ELSEIF ( trans .EQ. prgrd_max ) THEN
	           accum = arbitrary_small_val4
	         ENDIF
	         npts = 0

	         DO srcss = src_lo, src_hi

* ... ignore missing source data
	            IF ( src(i,j,k,srcss,m,n) .EQ. bad_src ) CYCLE

* ... contribution from this source box
	            IF     ( trans .EQ. prgrd_min ) THEN
	               accum = MIN( accum, src(i,j,k,srcss,m,n) )
	            ELSEIF ( trans .EQ. prgrd_max ) THEN
	               accum = MAX( accum, src(i,j,k,srcss,m,n) )
	            ENDIF
	            npts = npts + 1
                 ENDDO

                 IF ( npts .GT. 0 ) THEN
                    dst(i,j,k,dstss,m,n) = accum
                 ELSE
                    dst(i,j,k,dstss,m,n) = bad_dst
                 ENDIF

	      ENDDO

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* along E axis
        ELSEIF ( axis .EQ. E_dim ) THEN

           DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
           DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
           DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* loop through the each destination grid box
	      DO dstss = dstss_lo,dstss_hi

	         src_lo = sslo(dstss-dstss_lo+1)
	         src_hi = sshi(dstss-dstss_lo+1)

	         IF     ( trans .EQ. prgrd_min ) THEN
	            accum = arbitrary_large_val4
	         ELSEIF ( trans .EQ. prgrd_max ) THEN
	            accum = arbitrary_small_val4
	         ENDIF
	         npts = 0

	         DO srcss = src_lo, src_hi

* ... ignore missing source data
	            IF ( src(i,j,k,l,srcss,n) .EQ. bad_src ) CYCLE

* ... contribution from this source box
	            IF     ( trans .EQ. prgrd_min ) THEN
	               accum = MIN( accum, src(i,j,k,l,srcss,n) )
	            ELSEIF ( trans .EQ. prgrd_max ) THEN
	               accum = MAX( accum, src(i,j,k,l,srcss,n) )
	            ENDIF
	            npts = npts + 1
	         ENDDO

                 IF ( npts .GT. 0 ) THEN
                    dst(i,j,k,l,dstss,n) = accum
                 ELSE
                    dst(i,j,k,l,dstss,n) = bad_dst
                 ENDIF

	      ENDDO

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* along F axis
        ELSEIF ( axis .EQ. F_dim ) THEN

           DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
           DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
           DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* loop through the each destination grid box
	      DO dstss = dstss_lo,dstss_hi

	         src_lo = sslo(dstss-dstss_lo+1)
	         src_hi = sshi(dstss-dstss_lo+1)

	         IF     ( trans .EQ. prgrd_min ) THEN
	            accum = arbitrary_large_val4
	         ELSEIF ( trans .EQ. prgrd_max ) THEN
	            accum = arbitrary_small_val4
	         ENDIF
	         npts = 0

	         DO srcss = src_lo, src_hi

* ... ignore missing source data
	            IF ( src(i,j,k,l,m,srcss) .EQ. bad_src ) CYCLE

* ... contribution from this source box
	            IF     ( trans .EQ. prgrd_min ) THEN
	               accum = MIN( accum, src(i,j,k,l,m,srcss) )
	            ELSEIF ( trans .EQ. prgrd_max ) THEN
	               accum = MAX( accum, src(i,j,k,l,m,srcss) )
	            ENDIF
	            npts = npts + 1
	         ENDDO

                 IF ( npts .GT. 0 ) THEN
                    dst(i,j,k,l,m,dstss) = accum
                 ELSE
                    dst(i,j,k,l,m,dstss) = bad_dst
                 ENDIF

	      ENDDO

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

        ENDIF

	RETURN
	END
