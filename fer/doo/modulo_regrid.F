	SUBROUTINE MODULO_REGRID( cx_lims, axis, trans,
     .			src, msrc, dst, mdst, ngd, 
     .			mngd, nbd, mnbd, ave, mave)

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
* 	grid dst_grid by "modulo averaging".  Specifically, average all
*	of the source points that fall within the modulo-replications
*	of each destination point.

* 	msrc and mdst point to data structures in COMMON/XVARIABLES/ which
* 	specify the subscript bounds of src and dst within their 
* 	respective grids.

*	ss21 holds the indices of source grid boxes (on src_grid) for box
*	limits on dst_grid.

* 	programmer - Steve Hankin
* 	NOAA/PMEL,Seattle,WA - Tropical Modeling and Analysis Program

* V420:	 11/95
* V450:	 1/97 - bug fix: error in nfold calculation
* kob*   9/97 - replaced call to ISUBSCRIPT w/ call to ISUBSCR_CX in order
*               to have access to context information - needed for negative
*               time step processing
* V541:	 *sh*  2/02 - support for subspan modulo axes -- dont use line_dim 
* V561: *acm*  4/04 - (old fix got checked in, then an old version got checked in!!)
*                     add checks on source points when computing sum of squared deviations
*                     for MODVAR
* V603 5/07  *acm* - added MODNBD modulo regridding statistic
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
*       *acm* 4/12 6D Ferret: time axis may be in t or f direction. 
* V741  5/18 *acm* issue 1868: gridding transforms that will grid to same axis
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'		! for line_dim
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xcontext.cmn'
	include	'xmem_subsc.cmn'
	include 'xvariables.cmn'

* calling argument declarations:
	INTEGER	cx_lims, axis, trans, msrc, mdst, mngd, mnbd, mave

	REAL    src( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          dst( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
     .          ngd( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,
     .               m3lot:m3hit,m3loe:m3hie,m3lof:m3hif ),
     .          nbd( m4lox:m4hix,m4loy:m4hiy,m4loz:m4hiz,
     .               m4lot:m4hit,m4loe:m4hie,m4lof:m4hif ),
     .          ave( m5lox:m5hix,m5loy:m5hiy,m5loz:m5hiz,
     .               m5lot:m5hit,m5loe:m5hie,m5lof:m5hif )

* local variable declarations:
	LOGICAL	IS_SELF_REGRID, self_vari
	INTEGER	ISUBSCR_CX, TM_MODULO_LINE_DIM,
     .		dstss, srcss, src_grid, dst_grid,
     .		i, j, k, l, m, n, nmod, nfold, dst_start,
     .		srcss_lo, srcss_hi, dstss_lo, dstss_hi, status

	REAL	bad_src, bad_dst, tmp
	REAL*8	TDEST_WORLD, dstww

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

* Is this a regridding to the same line: Compute a result if @MODVAR on same axis,
* 0 for valid data, missing otherwise.

	self_vari = IS_SELF_REGRID(axis, src_grid, dst_grid, trans) .AND. 
     .           trans.EQ.prgrd_modvar

* initialize outputs to zero (ngd serves as flag for a valid calculation, too)
        DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
        DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
        DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
        DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
        DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
        DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
	   ngd(i,j,k,l,m,n) = 0.0
	   nbd(i,j,k,l,m,n) = 0.0

	   IF ( trans .EQ. prgrd_modulo ) THEN
	      dst(i,j,k,l,m,n) = 0.0
	   ELSEIF ( trans .EQ. prgrd_modsum ) THEN
	      dst(i,j,k,l,m,n) = 0.0
	   ELSEIF ( trans .EQ. prgrd_modvar ) THEN
	      dst(i,j,k,l,m,n) = 0.0
	      ave(i,j,k,l,m,n) = 0.0
	   ELSEIF ( trans .EQ. prgrd_modngd ) THEN
	      dst(i,j,k,l,m,n) = 0.0
	   ELSEIF ( trans .EQ. prgrd_modnbd ) THEN
	      dst(i,j,k,l,m,n) = 0.0
	   ELSEIF ( trans .EQ. prgrd_modmin ) THEN
	      dst(i,j,k,l,m,n) = arbitrary_large_val4
	   ELSEIF ( trans .EQ. prgrd_modmax ) THEN
	      dst(i,j,k,l,m,n) = arbitrary_small_val4
	   ENDIF

	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO

* the modulo index length of the destination axis
! ...	nmod = CAXIS_LEN(axis,cx_dst)	if cx_dst were known ...
	nmod = TM_MODULO_LINE_DIM( grid_line(axis,dst_grid) )

* the number of times the destination axis was folded before the first dstss_lo
!	nfold = INT(dstss_lo/nmod) * nmod
	nfold = INT((dstss_lo-1)/nmod) * nmod		! 1/97

*****************************************

* along X axis
        IF ( axis .EQ. x_dim ) THEN

* loop through the each source grid point
	   DO srcss = srcss_lo,srcss_hi

* what destination grid point owns this source point?
	      dstww = TDEST_WORLD(srcss, src_grid, axis, box_middle)
	      dstss = ISUBSCR_CX(dstww, dst_grid, axis, cx_lims,round_up)

* map that index, modulo-style, to the first target destination point
	      dst_start = MOD(dstss-1,nmod) + 1 + nfold
	      IF (dst_start .LT. dstss_lo) dst_start = dst_start + nmod

* add this contribution to all of the appropriate destination points
	      DO dstss = dst_start, dstss_hi, nmod
                 DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
                 DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
                 DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
                 DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
                 DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)

* ... ignore missing source data
	         IF ( src(srcss,j,k,l,m,n) .EQ. bad_src ) THEN
                    nbd(dstss,j,k,l,m,n) = nbd(dstss,j,k,l,m,n) + 1.
                    IF ( trans .EQ. prgrd_modnbd ) 
     .                  dst(dstss,j,k,l,m,n) = nbd(dstss,j,k,l,m,n)
                    CYCLE
                 ENDIF

* ... count this point
	         ngd(dstss,j,k,l,m,n) = ngd(dstss,j,k,l,m,n) + 1.

	         IF ( trans .EQ. prgrd_modulo ) THEN
	            dst(dstss,j,k,l,m,n) = dst(dstss,j,k,l,m,n) +
     .                                     src(srcss,j,k,l,m,n)
	         ELSEIF ( trans .EQ. prgrd_modsum ) THEN
	            dst(dstss,j,k,l,m,n) = dst(dstss,j,k,l,m,n) +
     .                                     src(srcss,j,k,l,m,n)
	         ELSEIF ( trans .EQ. prgrd_modvar ) THEN
	            ave(dstss,j,k,l,m,n) = ave(dstss,j,k,l,m,n) +
     .                                     src(srcss,j,k,l,m,n)
	         ELSEIF ( trans .EQ. prgrd_modngd ) THEN
	            dst(dstss,j,k,l,m,n) = ngd(dstss,j,k,l,m,n)
	         ELSEIF ( trans .EQ. prgrd_modmin ) THEN
	            IF (src(srcss,j,k,l,m,n) .LT. dst(dstss,j,k,l,m,n))
     .			dst(dstss,j,k,l,m,n) = src(srcss,j,k,l,m,n)
	         ELSEIF ( trans .EQ. prgrd_modmax ) THEN
	            IF (src(srcss,j,k,l,m,n) .GT. dst(dstss,j,k,l,m,n))
     .			dst(dstss,j,k,l,m,n) = src(srcss,j,k,l,m,n)
	         ENDIF

		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO

	      ENDDO

	   ENDDO

* along Y axis
        ELSEIF ( axis .EQ. Y_dim ) THEN

* loop through the each source grid point
	   DO srcss = srcss_lo,srcss_hi

* what destination grid point owns this source point?
	      dstww = TDEST_WORLD(srcss, src_grid, axis, box_middle)
	      dstss = ISUBSCR_CX(dstww, dst_grid, axis, cx_lims, round_up)

* map that index, modulo-style, to the first target destination point
	      dst_start = MOD(dstss-1,nmod) + 1 + nfold
	      IF (dst_start .LT. dstss_lo) dst_start = dst_start + nmod

* add this contribution to all of the appropriate destination points
	      DO dstss = dst_start, dstss_hi, nmod
                 DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
                 DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
                 DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
                 DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
                 DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* ... ignore missing source data
	         IF ( src(i,srcss,k,l,m,n) .EQ. bad_src ) THEN
                    nbd(i,dstss,k,l,m,n) = nbd(i,dstss,k,l,m,n) + 1.
                    IF ( trans .EQ. prgrd_modnbd ) 
     .                 dst(i,dstss,k,l,m,n) = nbd(i,dstss,k,l,m,n)
                    CYCLE
                 ENDIF

* ... count this point
	         ngd(i,dstss,k,l,m,n) = ngd(i,dstss,k,l,m,n) + 1.

	         IF ( trans .EQ. prgrd_modulo ) THEN
	            dst(i,dstss,k,l,m,n) = dst(i,dstss,k,l,m,n) +
     .                                     src(i,srcss,k,l,m,n)
	         ELSEIF ( trans .EQ. prgrd_modsum ) THEN
	            dst(i,dstss,k,l,m,n) = dst(i,dstss,k,l,m,n) +
     .                                     src(i,srcss,k,l,m,n)
	         ELSEIF ( trans .EQ. prgrd_modvar ) THEN
	            ave(i,dstss,k,l,m,n) = ave(i,dstss,k,l,m,n) +
     .                                     src(i,srcss,k,l,m,n)
	         ELSEIF ( trans .EQ. prgrd_modngd ) THEN
	            dst(i,dstss,k,l,m,n) = ngd(i,dstss,k,l,m,n)
	         ELSEIF ( trans .EQ. prgrd_modmin ) THEN
	            IF (src(i,srcss,k,l,m,n) .LT. dst(i,dstss,k,l,m,n))
     .			dst(i,dstss,k,l,m,n) = src(i,srcss,k,l,m,n)
	         ELSEIF ( trans .EQ. prgrd_modmax ) THEN
	            IF (src(i,srcss,k,l,m,n) .GT. dst(i,dstss,k,l,m,n))
     .			dst(i,dstss,k,l,m,n) = src(i,srcss,k,l,m,n)
	         ENDIF

		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO

	      ENDDO

	   ENDDO

* along Z axis
        ELSEIF ( axis .EQ. z_dim ) THEN

* loop through the each source grid point
	   DO srcss = srcss_lo,srcss_hi

* what destination grid point owns this source point?
	      dstww = TDEST_WORLD(srcss, src_grid, axis, box_middle)
	      dstss = ISUBSCR_CX(dstww, dst_grid, axis, cx_lims, round_up)

* map that index, modulo-style, to the first target destination point
	      dst_start = MOD(dstss-1,nmod) + 1 + nfold
	      IF (dst_start .LT. dstss_lo) dst_start = dst_start + nmod

* add this contribution to all of the appropriate destination points
	      DO dstss = dst_start, dstss_hi, nmod
                 DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
                 DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
                 DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
                 DO k = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
                 DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* ... ignore missing source data
	         IF ( src(i,j,srcss,l,m,n) .EQ. bad_src ) THEN
                    nbd(i,j,dstss,l,m,n) = nbd(i,j,dstss,l,m,n) + 1.
                    IF ( trans .EQ. prgrd_modnbd ) 
     .                  dst(i,j,dstss,l,m,n) = nbd(i,j,dstss,l,m,n)
                    CYCLE
                 ENDIF

* ... count this point
	         ngd(i,j,dstss,l,m,n) = ngd(i,j,dstss,l,m,n) + 1.

	         IF ( trans .EQ. prgrd_modulo ) THEN
	            dst(i,j,dstss,l,m,n) = dst(i,j,dstss,l,m,n) +
     .                                     src(i,j,srcss,l,m,n)
	         ELSEIF ( trans .EQ. prgrd_modsum ) THEN
	            dst(i,j,dstss,l,m,n) = dst(i,j,dstss,l,m,n) +
     .                                     src(i,j,srcss,l,m,n)
	         ELSEIF ( trans .EQ. prgrd_modvar ) THEN
	            ave(i,j,dstss,l,m,n) = ave(i,j,dstss,l,m,n) +
     .                                     src(i,j,srcss,l,m,n)
	         ELSEIF ( trans .EQ. prgrd_modngd ) THEN
	            dst(i,j,dstss,l,m,n) = ngd(i,j,dstss,l,m,n)
	         ELSEIF ( trans .EQ. prgrd_modmin ) THEN
	            IF (src(i,j,srcss,l,m,n) .LT. dst(i,j,dstss,l,m,n))
     .			dst(i,j,dstss,l,m,n) = src(i,j,srcss,l,m,n)
	         ELSEIF ( trans .EQ. prgrd_modmax ) THEN
	            IF (src(i,j,srcss,l,m,n) .GT. dst(i,j,dstss,l,m,n))
     .			dst(i,j,dstss,l,m,n) = src(i,j,srcss,l,m,n)
	         ENDIF

		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO

	      ENDDO

	   ENDDO

* along T axis
        ELSEIF ( axis .EQ. T_dim ) THEN

*   ==>   ALL TIME/DATE CALCULATIONS ARE DONE IN THE DEST AXIS ENCODING
           CALL TDEST_WORLD_INIT ( src_grid, dst_grid, axis, status ) ! src<->dst
           IF ( status .NE. ferr_ok ) STOP 'bogus TDEST_WORLD_INIT'

* loop through the each source grid point
	   DO srcss = srcss_lo,srcss_hi

* USE DESTINATION GRID TIME/DATE ENCODINGS FOR ALL CALCULATIONS
* what destination grid point owns this source point?
	      dstww = TDEST_WORLD(srcss, src_grid, axis, box_middle)
	      dstss = ISUBSCR_CX(dstww, dst_grid, axis, cx_lims,round_up)

* map that index, modulo-style, to the first target destination point
	      dst_start = MOD(dstss-1,nmod) + 1 + nfold
	      IF (dst_start .LT. dstss_lo) dst_start = dst_start + nmod

* add this contribution to all of the appropriate destination points
	      DO dstss = dst_start, dstss_hi, nmod
                 DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
                 DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
                 DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
                 DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
                 DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* ... ignore missing source data
	         IF ( src(i,j,k,srcss,m,n) .EQ. bad_src ) THEN
                    nbd(i,j,k,dstss,m,n) = nbd(i,j,k,dstss,m,n) + 1.
                    IF ( trans .EQ. prgrd_modnbd ) 
     .                  dst(i,j,k,dstss,m,n) = nbd(i,j,k,dstss,m,n)
                    CYCLE
                 ENDIF

* ... count this point
	         ngd(i,j,k,dstss,m,n) = ngd(i,j,k,dstss,m,n) + 1.

	         IF ( trans .EQ. prgrd_modulo ) THEN
	            dst(i,j,k,dstss,m,n) = dst(i,j,k,dstss,m,n) +
     .                                     src(i,j,k,srcss,m,n)
	         ELSEIF ( trans .EQ. prgrd_modsum ) THEN
	            dst(i,j,k,dstss,m,n) = dst(i,j,k,dstss,m,n) +
     .                                     src(i,j,k,srcss,m,n)
	         ELSEIF ( trans .EQ. prgrd_modvar ) THEN
	            ave(i,j,k,dstss,m,n) = ave(i,j,k,dstss,m,n) +
     .                                     src(i,j,k,srcss,m,n)
	         ELSEIF ( trans .EQ. prgrd_modngd ) THEN
	            dst(i,j,k,dstss,m,n) = ngd(i,j,k,dstss,m,n)
	         ELSEIF ( trans .EQ. prgrd_modmin ) THEN
	            IF (src(i,j,k,srcss,m,n) .LT. dst(i,j,k,dstss,m,n))
     .			dst(i,j,k,dstss,m,n) = src(i,j,k,srcss,m,n)
	         ELSEIF ( trans .EQ. prgrd_modmax ) THEN
	            IF (src(i,j,k,srcss,m,n) .GT. dst(i,j,k,dstss,m,n))
     .			dst(i,j,k,dstss,m,n) = src(i,j,k,srcss,m,n)
	         ENDIF

		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO

	      ENDDO

	   ENDDO


* along E axis
        ELSEIF ( axis .EQ. e_dim ) THEN

* loop through the each source grid point
	   DO srcss = srcss_lo,srcss_hi

* what destination grid point owns this source point?
	      dstww = TDEST_WORLD(srcss, src_grid, axis, box_middle)
	      dstss = ISUBSCR_CX(dstww, dst_grid, axis, cx_lims,round_up)

* map that index, modulo-style, to the first target destination point
	      dst_start = MOD(dstss-1,nmod) + 1 + nfold
	      IF (dst_start .LT. dstss_lo) dst_start = dst_start + nmod

* add this contribution to all of the appropriate destination points
	      DO dstss = dst_start, dstss_hi, nmod
                 DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
                 DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
                 DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
                 DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
                 DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* ... ignore missing source data
	         IF ( src(i,j,k,l,srcss,n) .EQ. bad_src ) THEN
                    nbd(i,j,k,l,dstss,n) = nbd(i,j,k,l,dstss,n) + 1.
                    IF ( trans .EQ. prgrd_modnbd ) 
     .                  dst(i,j,k,l,dstss,n) = nbd(i,j,k,l,dstss,n)
                    CYCLE
                 ENDIF

* ... count this point
	         ngd(i,j,k,l,dstss,n) = ngd(i,j,k,l,dstss,n) + 1.

	         IF ( trans .EQ. prgrd_modulo ) THEN
	            dst(i,j,k,l,dstss,n) = dst(i,j,k,l,dstss,n) +
     .                                     src(i,j,k,l,srcss,n)
	         ELSEIF ( trans .EQ. prgrd_modsum ) THEN
	            dst(i,j,k,l,dstss,n) = dst(i,j,k,l,dstss,n) +
     .                                     src(i,j,k,l,srcss,n)
	         ELSEIF ( trans .EQ. prgrd_modvar ) THEN
	            ave(i,j,k,l,dstss,n) = ave(i,j,k,l,dstss,n) +
     .                                     src(i,j,k,l,srcss,n)
	         ELSEIF ( trans .EQ. prgrd_modngd ) THEN
	            dst(i,j,k,l,dstss,n) = ngd(i,j,k,l,dstss,n)
	         ELSEIF ( trans .EQ. prgrd_modmin ) THEN
	            IF (src(i,j,k,l,srcss,n) .LT. dst(i,j,k,l,dstss,n))
     .			dst(i,j,k,l,dstss,n) = src(i,j,k,l,srcss,n)
	         ELSEIF ( trans .EQ. prgrd_modmax ) THEN
	            IF (src(i,j,k,l,srcss,n) .GT. dst(i,j,k,l,dstss,n))
     .			dst(i,j,k,l,dstss,n) = src(i,j,k,l,srcss,n)
	         ENDIF

		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO

	      ENDDO

	   ENDDO

* along F axis
        ELSEIF ( axis .EQ. F_dim ) THEN

*   ==>   ALL TIME/DATE CALCULATIONS ARE DONE IN THE DEST AXIS ENCODING
           CALL TDEST_WORLD_INIT ( src_grid, dst_grid, axis, status ) ! src<->dst
           IF ( status .NE. ferr_ok ) STOP 'bogus TDEST_WORLD_INIT'

* loop through the each source grid point
	   DO srcss = srcss_lo,srcss_hi

* USE DESTINATION GRID TIME/DATE ENCODINGS FOR ALL CALCULATIONS
* what destination grid point owns this source point?
	      dstww = TDEST_WORLD(srcss, src_grid, axis, box_middle)
	      dstss = ISUBSCR_CX(dstww, dst_grid, axis, cx_lims,round_up)

* map that index, modulo-style, to the first target destination point
	      dst_start = MOD(dstss-1,nmod) + 1 + nfold
	      IF (dst_start .LT. dstss_lo) dst_start = dst_start + nmod

* add this contribution to all of the appropriate destination points
	      DO dstss = dst_start, dstss_hi, nmod
                 DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
                 DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
                 DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
                 DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
                 DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* ... ignore missing source data
	         IF ( src(i,j,k,l,m,srcss) .EQ. bad_src ) THEN
                    nbd(i,j,k,l,m,dstss) = nbd(i,j,k,l,m,dstss) + 1.
                    IF ( trans .EQ. prgrd_modnbd ) 
     .                  dst(i,j,k,l,m,dstss) = nbd(i,j,k,l,m,dstss)
                    CYCLE
                 ENDIF

* ... count this point
	         ngd(i,j,k,l,m,dstss) = ngd(i,j,k,l,m,dstss) + 1.

	         IF ( trans .EQ. prgrd_modulo ) THEN
	            dst(i,j,k,l,m,dstss) = dst(i,j,k,l,m,dstss) +
     .                                     src(i,j,k,l,m,srcss)
	         ELSEIF ( trans .EQ. prgrd_modsum ) THEN
	            dst(i,j,k,l,m,dstss) = dst(i,j,k,l,m,dstss) +
     .                                     src(i,j,k,l,m,srcss)
	         ELSEIF ( trans .EQ. prgrd_modvar ) THEN
	            ave(i,j,k,l,m,dstss) = ave(i,j,k,l,m,dstss) +
     .                                     src(i,j,k,l,m,srcss)
	         ELSEIF ( trans .EQ. prgrd_modngd ) THEN
	            dst(i,j,k,l,m,dstss) = ngd(i,j,k,l,m,dstss)
	         ELSEIF ( trans .EQ. prgrd_modmin ) THEN
	            IF (src(i,j,k,l,m,srcss) .LT. dst(i,j,k,l,m,dstss))
     .			dst(i,j,k,l,m,dstss) = src(i,j,k,l,m,srcss)
	         ELSEIF ( trans .EQ. prgrd_modmax ) THEN
	            IF (src(i,j,k,l,m,srcss) .GT. dst(i,j,k,l,m,dstss))
     .			dst(i,j,k,l,m,dstss) = src(i,j,k,l,m,srcss)
	         ENDIF

		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO

	      ENDDO

	   ENDDO

	ENDIF

	IF ( trans .EQ. prgrd_modngd ) RETURN
	IF ( trans .EQ. prgrd_modnbd ) RETURN

* for all axis orientations
* ... fill in missing values where there were zero source points
        DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
        DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
        DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
        DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
        DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
        DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

	   IF ( ngd(i,j,k,l,m,n) .EQ. 0.0 ) THEN
	      dst(i,j,k,l,m,n) = bad_dst
	   ELSEIF ( trans .EQ. prgrd_modulo ) THEN
	      dst(i,j,k,l,m,n) = dst(i,j,k,l,m,n) / ngd(i,j,k,l,m,n)
	   ELSEIF ( trans .EQ. prgrd_modvar ) THEN
	      ave(i,j,k,l,m,n) = ave(i,j,k,l,m,n) / ngd(i,j,k,l,m,n)
	      IF (self_vari) THEN
	         IF (ngd(i,j,k,l,m,n) .EQ. 1.) THEN 
		    dst(i,j,k,l,m,n) = 0.
		 ELSE
		    dst(i,j,k,l,m,n) = bad_dst
		 ENDIF
	      ENDIF
	   ENDIF

	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO

	IF ( trans .NE. prgrd_modvar ) RETURN
	IF ( self_vari ) RETURN

**********************

* PASS #2 (needed only to compute variance - @MODVAR)
* ... at this point ngd contains the number of points
*     and ave contains the averages

* along X axis
        IF ( axis .EQ. x_dim ) THEN

* loop through the each source grid point
	   DO srcss = srcss_lo,srcss_hi

* what destination grid point owns this source point?
	      dstww = TDEST_WORLD(srcss, src_grid, axis, box_middle)
	      dstss = ISUBSCR_CX(dstww, dst_grid, axis, cx_lims, round_up)

* map that index, modulo-style, to the first target destination point
	      dst_start = MOD(dstss-1,nmod) + 1 + nfold
	      IF (dst_start .LT. dstss_lo) dst_start = dst_start + nmod

* add this contribution to all of the appropriate destination points
	      DO dstss = dst_start, dstss_hi, nmod
                 DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
                 DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
                 DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
                 DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
                 DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)

* sum of squared deviations
	         IF ( ngd(dstss,j,k,l,m,n) .GT. 1.0 ) THEN
* ... must be at least 2 source points to get a variance
		    IF (src(srcss,j,k,l,m,n) .NE. bad_src) THEN
	               tmp = src(srcss,j,k,l,m,n)-ave(dstss,j,k,l,m,n)
	               dst(dstss,j,k,l,m,n) = dst(dstss,j,k,l,m,n) +
     .                                        tmp * tmp
                    ENDIF
	         ENDIF

		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO

	      ENDDO

	   ENDDO


* along Y axis
        ELSEIF ( axis .EQ. Y_dim ) THEN

* loop through the each source grid point
	   DO srcss = srcss_lo,srcss_hi

* what destination grid point owns this source point?
	      dstww = TDEST_WORLD(srcss, src_grid, axis, box_middle)
	      dstss = ISUBSCR_CX(dstww, dst_grid, axis, cx_lims,round_up)

* map that index, modulo-style, to the first target destination point
	      dst_start = MOD(dstss-1,nmod) + 1 + nfold
	      IF (dst_start .LT. dstss_lo) dst_start = dst_start + nmod

* add this contribution to all of the appropriate destination points
	      DO dstss = dst_start, dstss_hi, nmod
                 DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
                 DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
                 DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
                 DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
                 DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* sum of squared deviations
	         IF ( ngd(i,dstss,k,l,m,n) .GT. 1.0 ) THEN
* ... must be at least 2 source points to get a variance
		    IF (src(i,srcss,k,l,m,n) .NE. bad_src) THEN
		        tmp = src(i,srcss,k,l,m,n)-ave(i,dstss,k,l,m,n)
		        dst(i,dstss,k,l,m,n) = dst(i,dstss,k,l,m,n) +
     .                                         tmp * tmp
		    ENDIF
	         ENDIF

		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO

	      ENDDO

	   ENDDO

* along Z axis
        ELSEIF ( axis .EQ. z_dim ) THEN

* loop through the each source grid point
	   DO srcss = srcss_lo,srcss_hi

* what destination grid point owns this source point?
	      dstww = TDEST_WORLD(srcss, src_grid, axis, box_middle)
	      dstss = ISUBSCR_CX(dstww, dst_grid, axis, cx_lims, round_up)

* map that index, modulo-style, to the first target destination point
	      dst_start = MOD(dstss-1,nmod) + 1 + nfold
	      IF (dst_start .LT. dstss_lo) dst_start = dst_start + nmod

* add this contribution to all of the appropriate destination points
	      DO dstss = dst_start, dstss_hi, nmod
                 DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
                 DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
                 DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
                 DO k = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
                 DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* sum of squared deviations
	         IF ( ngd(i,j,dstss,l,m,n) .GT. 1.0 ) THEN
* ... must be at least 2 source points to get a variance
		    IF (src(i,j,srcss,l,m,n) .NE. bad_src) THEN
		       tmp = src(i,j,srcss,l,m,n)-ave(i,j,dstss,l,m,n)
		       dst(i,j,dstss,l,m,n) = dst(i,j,dstss,l,m,n) +
     .                                        tmp * tmp
		    ENDIF
	         ENDIF

		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO

	      ENDDO

	   ENDDO


* along T axis
        ELSEIF (  axis .EQ. T_dim ) THEN

* loop through the each source grid point
	   DO srcss = srcss_lo,srcss_hi

* USE DESTINATION GRID TIME/DATE ENCODINGS FOR ALL CALCULATIONS
* what destination grid point owns this source point?
	      dstww = TDEST_WORLD(srcss, src_grid, axis, box_middle)
	      dstss = ISUBSCR_CX(dstww, dst_grid, axis, cx_lims, round_up)

* map that index, modulo-style, to the first target destination point
	      dst_start = MOD(dstss-1,nmod) + 1 + nfold
	      IF (dst_start .LT. dstss_lo) dst_start = dst_start + nmod

* add this contribution to all of the appropriate destination points
	      DO dstss = dst_start, dstss_hi, nmod
                 DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
                 DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
                 DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
                 DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
                 DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* sum of squared deviations
	         IF ( ngd(i,j,k,dstss,m,n) .GT. 1.0 ) THEN
* ... must be at least 2 source points to get a variance
		    IF (src(i,j,k,srcss,m,n) .NE. bad_src) THEN
		       tmp = src(i,j,k,srcss,m,n)-ave(i,j,k,dstss,m,n)
		       dst(i,j,k,dstss,m,n) = dst(i,j,k,dstss,m,n) +
     .                                        tmp * tmp
		    ENDIF
	         ENDIF

		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO

	      ENDDO

	   ENDDO
 

* along E axis
        ELSEIF ( axis .EQ. e_dim ) THEN

* loop through the each source grid point
	   DO srcss = srcss_lo,srcss_hi

* what destination grid point owns this source point?
	      dstww = TDEST_WORLD(srcss, src_grid, axis, box_middle)
	      dstss = ISUBSCR_CX(dstww, dst_grid, axis, cx_lims, round_up)

* map that index, modulo-style, to the first target destination point
	      dst_start = MOD(dstss-1,nmod) + 1 + nfold
	      IF (dst_start .LT. dstss_lo) dst_start = dst_start + nmod

* add this contribution to all of the appropriate destination points
	      DO dstss = dst_start, dstss_hi, nmod
                 DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
                 DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
                 DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
                 DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
                 DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* sum of squared deviations
	         IF ( ngd(i,j,k,l,dstss,n) .GT. 1.0 ) THEN
* ... must be at least 2 source points to get a variance
		    IF (src(i,j,k,l,srcss,n) .NE. bad_src) THEN
	               tmp = src(i,j,k,l,srcss,n)-ave(i,j,k,l,dstss,n)
	               dst(i,j,k,l,dstss,n) = dst(i,j,k,l,dstss,n) +
     .                                        tmp * tmp
                    ENDIF
	         ENDIF

		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO

	      ENDDO

	   ENDDO

* along F axis
        ELSEIF (  axis .EQ. f_dim ) THEN

* loop through the each source grid point
	   DO srcss = srcss_lo,srcss_hi

* USE DESTINATION GRID TIME/DATE ENCODINGS FOR ALL CALCULATIONS
* what destination grid point owns this source point?
	      dstww = TDEST_WORLD(srcss, src_grid, axis, box_middle)
	      dstss = ISUBSCR_CX(dstww, dst_grid, axis, cx_lims, round_up)

* map that index, modulo-style, to the first target destination point
	      dst_start = MOD(dstss-1,nmod) + 1 + nfold
	      IF (dst_start .LT. dstss_lo) dst_start = dst_start + nmod

* add this contribution to all of the appropriate destination points
	      DO dstss = dst_start, dstss_hi, nmod
                 DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
                 DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
                 DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
                 DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
                 DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* sum of squared deviations
	         IF ( ngd(i,j,k,l,m,dstss) .GT. 1.0 ) THEN
* ... must be at least 2 source points to get a variance
		    IF (src(i,j,k,l,m,srcss) .NE. bad_src) THEN
		       tmp = src(i,j,k,l,m,srcss)-ave(i,j,k,l,m,dstss)
		       dst(i,j,k,l,m,dstss) = dst(i,j,k,l,m,dstss) +
     .                                        tmp * tmp
		    ENDIF
	         ENDIF

		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO
		 ENDDO

	      ENDDO

	   ENDDO

	ENDIF

* for all axis orientations: divide sum of squared deviations by N-1
        DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
        DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
        DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
        DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
        DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
        DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

	   IF ( ngd(i,j,k,l,m,n) .LE. 1.0 ) THEN
	      dst(i,j,k,l,m,n) = bad_dst
	   ELSE
	      dst(i,j,k,l,m,n) = dst(i,j,k,l,m,n) /
     .                           (ngd(i,j,k,l,m,n) - 1.0)
	   ENDIF

	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO

	RETURN
	END
