	SUBROUTINE MODULO_REGRID( cx_lims, axis, trans,
     .			src, msrc, dst, mdst, ngd, mngd, ave, mave)

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

	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'		! for line_dim
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xcontext.cmn'
	include	'xmem_subsc.cmn'
	include 'xvariables.cmn'

* calling argument declarations:
	INTEGER	cx_lims, axis, trans, msrc, mdst, mngd, mave

	REAL    src( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		dst( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		ngd( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit ),
     .		ave( m4lox:m4hix,m4loy:m4hiy,m4loz:m4hiz,m4lot:m4hit )

* local variable declarations:
	INTEGER	ISUBSCR_CX, TM_MODULO_LINE_DIM,
     .		dstss, srcss, src_grid, dst_grid,
     .		i, j, k, l, nmod, nfold, dst_start,
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

* initialize outputs to zero (ngd serves as flag for a valid calculation, too)
        DO 10 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
        DO 10 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
        DO 10 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
        DO 10 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
	   ngd(i, j, k, l) = 0.0

	   IF ( trans .EQ. prgrd_modulo ) THEN
	      dst(i,j,k,l) = 0.0
	   ELSEIF ( trans .EQ. prgrd_modsum ) THEN
	      dst(i,j,k,l) = 0.0
	   ELSEIF ( trans .EQ. prgrd_modvar ) THEN
	      dst(i,j,k,l) = 0.0
	      ave(i,j,k,l) = 0.0
	   ELSEIF ( trans .EQ. prgrd_modngd ) THEN
	      dst(i,j,k,l) = 0.0
	   ELSEIF ( trans .EQ. prgrd_modmin ) THEN
	      dst(i,j,k,l) = arbitrary_large_val4
	   ELSEIF ( trans .EQ. prgrd_modmax ) THEN
	      dst(i,j,k,l) = arbitrary_small_val4
	   ENDIF

 10	CONTINUE
	
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
	   DO 190 srcss = srcss_lo,srcss_hi

* what destination grid point owns this source point?
	      dstww = TDEST_WORLD(srcss, src_grid, axis, box_middle )
	      dstss = ISUBSCR_CX(dstww, dst_grid, axis, cx_lims,round_up )

* map that index, modulo-style, to the first target destination point
	      dst_start = MOD(dstss-1,nmod)+1 + nfold
	      IF (dst_start .LT. dstss_lo) dst_start = dst_start + nmod

* add this contribution to all of the appropriate destination points
	      DO 120 dstss = dst_start, dstss_hi, nmod
                 DO 110 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
                 DO 110 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
                 DO 110 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)

* ... ignore missing source data
	         IF ( src( srcss,j,k,l ) .EQ. bad_src ) GOTO 110

* ... count this point
	         ngd(dstss,j,k,l) = ngd(dstss,j,k,l) + 1.

	         IF ( trans .EQ. prgrd_modulo ) THEN
	            dst(dstss,j,k,l) = dst(dstss,j,k,l) + src(srcss,j,k,l)
	         ELSEIF ( trans .EQ. prgrd_modsum ) THEN
	            dst(dstss,j,k,l) = dst(dstss,j,k,l) + src(srcss,j,k,l)
	         ELSEIF ( trans .EQ. prgrd_modvar ) THEN
	            ave(dstss,j,k,l) = ave(dstss,j,k,l) + src(srcss,j,k,l)
	         ELSEIF ( trans .EQ. prgrd_modngd ) THEN
	            dst(dstss,j,k,l) = ngd(dstss,j,k,l)
	         ELSEIF ( trans .EQ. prgrd_modmin ) THEN
	            IF ( src(srcss,j,k,l) .LT. dst(dstss,j,k,l) )
     .			dst(dstss,j,k,l) = src(srcss,j,k,l)
	         ELSEIF ( trans .EQ. prgrd_modmax ) THEN
	            IF ( src(srcss,j,k,l) .GT. dst(dstss,j,k,l) )
     .			dst(dstss,j,k,l) = src(srcss,j,k,l)
	         ENDIF

 110	         CONTINUE
 120	      CONTINUE

 190	   CONTINUE

* along Y axis
        ELSEIF ( axis .EQ. Y_dim ) THEN

* loop through the each source grid point
	   DO 290 srcss = srcss_lo,srcss_hi

* what destination grid point owns this source point?
	      dstww = TDEST_WORLD(srcss, src_grid, axis, box_middle )
	      dstss = ISUBSCR_CX(dstww, dst_grid, axis, cx_lims, round_up )

* map that index, modulo-style, to the first target destination point
	      dst_start = MOD(dstss-1,nmod)+1 + nfold
	      IF (dst_start .LT. dstss_lo) dst_start = dst_start + nmod

* add this contribution to all of the appropriate destination points
	      DO 220 dstss = dst_start, dstss_hi, nmod
                 DO 210 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
                 DO 210 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
                 DO 210 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* ... ignore missing source data
	         IF ( src( i,srcss,k,l ) .EQ. bad_src ) GOTO 210

* ... count this point
	         ngd(i,dstss,k,l) = ngd(i,dstss,k,l) + 1.

	         IF ( trans .EQ. prgrd_modulo ) THEN
	            dst(i,dstss,k,l) = dst(i,dstss,k,l) + src(i,srcss,k,l)
	         ELSEIF ( trans .EQ. prgrd_modsum ) THEN
	            dst(i,dstss,k,l) = dst(i,dstss,k,l) + src(i,srcss,k,l)
	         ELSEIF ( trans .EQ. prgrd_modvar ) THEN
	            ave(i,dstss,k,l) = ave(i,dstss,k,l) + src(i,srcss,k,l)
	         ELSEIF ( trans .EQ. prgrd_modngd ) THEN
	            dst(i,dstss,k,l) = ngd(i,dstss,k,l)
	         ELSEIF ( trans .EQ. prgrd_modmin ) THEN
	            IF ( src(i,srcss,k,l) .LT. dst(i,dstss,k,l) )
     .			dst(i,dstss,k,l) = src(i,srcss,k,l)
	         ELSEIF ( trans .EQ. prgrd_modmax ) THEN
	            IF ( src(i,srcss,k,l) .GT. dst(i,dstss,k,l) )
     .			dst(i,dstss,k,l) = src(i,srcss,k,l)
	         ENDIF

 210	         CONTINUE
 220	      CONTINUE

 290	   CONTINUE

* along Z axis
        ELSEIF ( axis .EQ. z_dim ) THEN

* loop through the each source grid point
	   DO 390 srcss = srcss_lo,srcss_hi

* what destination grid point owns this source point?
	      dstww = TDEST_WORLD(srcss, src_grid, axis, box_middle )
	      dstss = ISUBSCR_CX(dstww, dst_grid, axis, cx_lims, round_up )

* map that index, modulo-style, to the first target destination point
	      dst_start = MOD(dstss-1,nmod)+1 + nfold
	      IF (dst_start .LT. dstss_lo) dst_start = dst_start + nmod

* add this contribution to all of the appropriate destination points
	      DO 320 dstss = dst_start, dstss_hi, nmod
                 DO 310 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
                 DO 310 k = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
                 DO 310 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* ... ignore missing source data
	         IF ( src( i,j,srcss,l ) .EQ. bad_src ) GOTO 310

* ... count this point
	         ngd(i,j,dstss,l) = ngd(i,j,dstss,l) + 1.

	         IF ( trans .EQ. prgrd_modulo ) THEN
	            dst(i,j,dstss,l) = dst(i,j,dstss,l) + src(i,j,srcss,l)
	         ELSEIF ( trans .EQ. prgrd_modsum ) THEN
	            dst(i,j,dstss,l) = dst(i,j,dstss,l) + src(i,j,srcss,l)
	         ELSEIF ( trans .EQ. prgrd_modvar ) THEN
	            ave(i,j,dstss,l) = ave(i,j,dstss,l) + src(i,j,srcss,l)
	         ELSEIF ( trans .EQ. prgrd_modngd ) THEN
	            dst(i,j,dstss,l) = ngd(i,j,dstss,l)
	         ELSEIF ( trans .EQ. prgrd_modmin ) THEN
	            IF ( src(i,j,srcss,l) .LT. dst(i,j,dstss,l) )
     .			dst(i,j,dstss,l) = src(i,j,srcss,l)
	         ELSEIF ( trans .EQ. prgrd_modmax ) THEN
	            IF ( src(i,j,srcss,l) .GT. dst(i,j,dstss,l) )
     .			dst(i,j,dstss,l) = src(i,j,srcss,l)
	         ENDIF

 310	         CONTINUE
 320	      CONTINUE

 390	   CONTINUE

* along T axis
        ELSEIF ( axis .EQ. T_dim ) THEN

*   ==>   ALL TIME/DATE CALCULATIONS ARE DONE IN THE DEST AXIS ENCODING
           CALL TDEST_WORLD_INIT ( src_grid, dst_grid, status ) ! src<->dst
           IF ( status .NE. ferr_ok ) STOP 'bogus TDEST_WORLD_INIT'

* loop through the each source grid point
	   DO 490 srcss = srcss_lo,srcss_hi

* USE DESTINATION GRID TIME/DATE ENCODINGS FOR ALL CALCULATIONS
* what destination grid point owns this source point?
	      dstww = TDEST_WORLD(srcss, src_grid, axis, box_middle )
	      dstss = ISUBSCR_CX(dstww, dst_grid, axis, cx_lims,round_up )

* map that index, modulo-style, to the first target destination point
	      dst_start = MOD(dstss-1,nmod)+1 + nfold
	      IF (dst_start .LT. dstss_lo) dst_start = dst_start + nmod

* add this contribution to all of the appropriate destination points
	      DO 420 dstss = dst_start, dstss_hi, nmod
                 DO 410 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
                 DO 410 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
                 DO 410 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* ... ignore missing source data
	         IF ( src( i,j,k,srcss ) .EQ. bad_src ) GOTO 410

* ... count this point
	         ngd(i,j,k,dstss) = ngd(i,j,k,dstss) + 1.

	         IF ( trans .EQ. prgrd_modulo ) THEN
	            dst(i,j,k,dstss) = dst(i,j,k,dstss) + src(i,j,k,srcss)
	         ELSEIF ( trans .EQ. prgrd_modsum ) THEN
	            dst(i,j,k,dstss) = dst(i,j,k,dstss) + src(i,j,k,srcss)
	         ELSEIF ( trans .EQ. prgrd_modvar ) THEN
	            ave(i,j,k,dstss) = ave(i,j,k,dstss) + src(i,j,k,srcss)
	         ELSEIF ( trans .EQ. prgrd_modngd ) THEN
	            dst(i,j,k,dstss) = ngd(i,j,k,dstss)
	         ELSEIF ( trans .EQ. prgrd_modmin ) THEN
	            IF ( src(i,j,k,srcss) .LT. dst(i,j,k,dstss) )
     .			dst(i,j,k,dstss) = src(i,j,k,srcss)
	         ELSEIF ( trans .EQ. prgrd_modmax ) THEN
	            IF ( src(i,j,k,srcss) .GT. dst(i,j,k,dstss) )
     .			dst(i,j,k,dstss) = src(i,j,k,srcss)
	         ENDIF

 410	         CONTINUE
 420	      CONTINUE

 490	   CONTINUE
	ENDIF

	IF ( trans .EQ. prgrd_modngd ) RETURN

* for all axis orientations
* ... fill in missing values where there were zero source points
        DO 1000 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
        DO 1000 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
        DO 1000 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
        DO 1000 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

	   IF ( ngd(i, j, k, l) .EQ. 0.0 ) THEN
	      dst(i, j, k, l) = bad_dst
	   ELSEIF ( trans .EQ. prgrd_modulo ) THEN
	      dst(i, j, k, l) = dst(i, j, k, l) / ngd(i, j, k, l)
	   ELSEIF ( trans .EQ. prgrd_modvar ) THEN
	      ave(i, j, k, l) = ave(i, j, k, l) / ngd(i, j, k, l)
	   ENDIF

 1000	CONTINUE

	IF ( trans .NE. prgrd_modvar ) RETURN

**********************

* PASS #2 (needed only to compute variance - @MODVAR)
* ... at this point ngd contains the number of points
*     and ave contains the averages

* along X axis
        IF ( axis .EQ. x_dim ) THEN

* loop through the each source grid point
	   DO 1190 srcss = srcss_lo,srcss_hi

* what destination grid point owns this source point?
	      dstww = TDEST_WORLD(srcss, src_grid, axis, box_middle )
	      dstss = ISUBSCR_CX(dstww, dst_grid, axis, cx_lims, round_up )

* map that index, modulo-style, to the first target destination point
	      dst_start = MOD(dstss-1,nmod)+1 + nfold
	      IF (dst_start .LT. dstss_lo) dst_start = dst_start + nmod

* add this contribution to all of the appropriate destination points
	      DO 1120 dstss = dst_start, dstss_hi, nmod
                 DO 1110 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
                 DO 1110 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
                 DO 1110 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)

* sum of squared deviations
	         IF ( ngd(dstss,j,k,l) .GT. 1.0 ) THEN
* ... must be at least 2 source points to get a variance
	            tmp = src(srcss,j,k,l)- ave(dstss, j, k, l)
	            dst(dstss,j,k,l) = dst(dstss,j,k,l) + tmp*tmp
	         ENDIF

 1110	         CONTINUE
 1120	      CONTINUE

 1190	   CONTINUE


* along Y axis
        ELSEIF ( axis .EQ. Y_dim ) THEN

* loop through the each source grid point
	   DO 1290 srcss = srcss_lo,srcss_hi

* what destination grid point owns this source point?
	      dstww = TDEST_WORLD(srcss, src_grid, axis, box_middle )
	      dstss = ISUBSCR_CX(dstww, dst_grid, axis, cx_lims,round_up )

* map that index, modulo-style, to the first target destination point
	      dst_start = MOD(dstss-1,nmod)+1 + nfold
	      IF (dst_start .LT. dstss_lo) dst_start = dst_start + nmod

* add this contribution to all of the appropriate destination points
	      DO 1220 dstss = dst_start, dstss_hi, nmod
                 DO 1210 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
                 DO 1210 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
                 DO 1210 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* sum of squared deviations
	         IF ( ngd(i,dstss,k,l) .GT. 1.0 ) THEN
* ... must be at least 2 source points to get a variance
	            tmp = src(i,srcss,k,l)- ave(i, dstss, k, l)
	            dst(i,dstss,k,l) = dst(i,dstss,k,l) + tmp*tmp
	         ENDIF

 1210	         CONTINUE
 1220	      CONTINUE

 1290	   CONTINUE


* along Z axis
        ELSEIF ( axis .EQ. z_dim ) THEN

* loop through the each source grid point
	   DO 1390 srcss = srcss_lo,srcss_hi

* what destination grid point owns this source point?
	      dstww = TDEST_WORLD(srcss, src_grid, axis, box_middle )
	      dstss = ISUBSCR_CX(dstww, dst_grid, axis, cx_lims, round_up )

* map that index, modulo-style, to the first target destination point
	      dst_start = MOD(dstss-1,nmod)+1 + nfold
	      IF (dst_start .LT. dstss_lo) dst_start = dst_start + nmod

* add this contribution to all of the appropriate destination points
	      DO 1320 dstss = dst_start, dstss_hi, nmod
                 DO 1310 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
                 DO 1310 k = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
                 DO 1310 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* sum of squared deviations
	         IF ( ngd(i,j,dstss,l) .GT. 1.0 ) THEN
* ... must be at least 2 source points to get a variance
	            tmp = src(i,j,srcss,l)- ave(i, j, dstss, l)
	            dst(i,j,dstss,l) = dst(i,j,dstss,l) + tmp*tmp
	         ENDIF

 1310	         CONTINUE
 1320	      CONTINUE

 1390	   CONTINUE


* along T axis
        ELSEIF ( axis .EQ. T_dim ) THEN

* loop through the each source grid point
	   DO 1490 srcss = srcss_lo,srcss_hi

* USE DESTINATION GRID TIME/DATE ENCODINGS FOR ALL CALCULATIONS
* what destination grid point owns this source point?
	      dstww = TDEST_WORLD(srcss, src_grid, axis, box_middle )
	      dstss = ISUBSCR_CX(dstww, dst_grid, axis, cx_lims, round_up )

* map that index, modulo-style, to the first target destination point
	      dst_start = MOD(dstss-1,nmod)+1 + nfold
	      IF (dst_start .LT. dstss_lo) dst_start = dst_start + nmod

* add this contribution to all of the appropriate destination points
	      DO 1420 dstss = dst_start, dstss_hi, nmod
                 DO 1410 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
                 DO 1410 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
                 DO 1410 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* sum of squared deviations
	         IF ( ngd(i,j,k,dstss) .GT. 1.0 ) THEN
* ... must be at least 2 source points to get a variance
	            tmp = src(i,j,k,srcss)- ave(i, j, k, dstss)
	            dst(i,j,k,dstss) = dst(i,j,k,dstss) + tmp*tmp
	         ENDIF

 1410	         CONTINUE
 1420	      CONTINUE

 1490	   CONTINUE
	ENDIF

* for all axis orientations: divide sum of squared deviations by N-1
        DO 2000 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
        DO 2000 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
        DO 2000 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
        DO 2000 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

	   IF ( ngd(i, j, k, l) .LE. 1.0 ) THEN
	      dst(i,j,k,l) = bad_dst
	   ELSE
	      dst(i,j,k,l) = dst(i,j,k,l) / (ngd(i,j,k,l) - 1.0)
	   ENDIF

 2000	CONTINUE

	RETURN
	END
