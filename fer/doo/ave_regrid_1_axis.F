	SUBROUTINE AVE_REGRID_1_AXIS( cx_lims, axis, trans,
     .				 dst_lox, dst_hix, dst_loy, dst_hiy,
     .				 src, msrc, dst, mdst, ss21, wksize )

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

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER	cx_lims, axis, trans, msrc, mdst,
     .          dst_lox, dst_hix, dst_loy, dst_hiy , wksize
	INTEGER	ss21(wksize)

	REAL    src( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		dst( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit )

* local variable declarations:
	INTEGER		dstss,srcss,src_grid,dst_grid,i,j,k,l
	INTEGER	     	srcss_lo,srcss_hi, dstss_lo,dstss_hi

	REAL		ddist, ydist, bad_src, bad_dst
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
           DO 190 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO 190 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO 190 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)

* loop through the each destination grid box
	   DO 160 dstss = dstss_lo,dstss_hi
	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

*  check if requested region is out of source range
              IF ( srcss_lo .EQ. unspecified_int4
     .	     .OR.  srcss_hi .EQ. unspecified_int4 ) THEN
	         dst(dstss,j,k,l) = bad_dst
	         goto 160
	      ENDIF

	      xdst_lo = TM_WORLD(dstss, dst_grid, axis, box_lo_lim )
	      xdst_hi = TM_WORLD(dstss, dst_grid, axis, box_hi_lim )

* add all partial or complete boxes of data from the source grid
	      sum = 0.0D0
	      distance = 0.0D0
	      DO 110 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src( srcss, j, k, l ) .EQ. bad_src ) GOTO 110

	         xsrc_min = TM_WORLD(srcss,src_grid,axis,box_lo_lim)
	         xsrc_max = TM_WORLD(srcss,src_grid,axis,box_hi_lim)

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi )

	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box
	         distance  = distance + ddist
	         IF (trans .EQ. prgrd_sum ) THEN
	            sum   = sum  + src(srcss, j, k, l)
     .			         * ddist/(xsrc_max-xsrc_min)
	         ELSEIF (trans .EQ. prgrd_good_pt ) THEN
	            sum   = sum  + ddist/(xsrc_max-xsrc_min)

	         ELSE	! average or variance
	            sum   = sum  + ddist * src(srcss, j, k, l)
	         ENDIF

 110	      CONTINUE

	      IF ( trans.EQ.prgrd_good_pt ) THEN
	         dst(dstss, j, k, l) = sum
	      ELSEIF ( distance .EQ. 0.0 ) THEN
	         dst(dstss, j, k, l) = bad_dst
	      ELSEIF ( trans.EQ.prgrd_sum ) THEN
	         dst(dstss, j, k, l) = sum
	      ELSE
* ... divide by distance to get average
	         dst(dstss, j, k, l) = sum / distance
	      ENDIF


 160	   CONTINUE
 190	   CONTINUE

* along Y axis
        ELSEIF ( axis .EQ. Y_dim ) THEN

           DO 290 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO 290 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO 290 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* loop through the each destination grid box
	   DO 260 dstss = dstss_lo,dstss_hi
	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

*  check if requested region is out of source range
              IF ( srcss_lo .EQ. unspecified_int4
     .	     .OR.  srcss_hi .EQ. unspecified_int4 ) THEN
	         dst(i,dstss,k,l) = bad_dst
	         goto 260
	      ENDIF

	      xdst_lo  = TM_WORLD(dstss, dst_grid, axis, box_lo_lim )
	      xdst_hi  = TM_WORLD(dstss, dst_grid, axis, box_hi_lim )

* add all partial or complete boxes of data from the source grid
	      sum = 0.0D0
	      distance = 0.0D0
	      DO 210 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src( i, srcss, k, l ) .EQ. bad_src ) GOTO 210

	         xsrc_min = TM_WORLD(srcss,src_grid,axis,box_lo_lim)
	         xsrc_max = TM_WORLD(srcss,src_grid,axis,box_hi_lim)

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi)

	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box
	         distance  = distance + ddist
	         IF (trans .EQ. prgrd_sum ) THEN
	            sum   = sum  + src(i, srcss, k, l)
     .			         * ddist/(xsrc_max-xsrc_min)
	         ELSEIF (trans .EQ. prgrd_good_pt ) THEN
	            sum   = sum  + ddist/(xsrc_max-xsrc_min)

	         ELSE	! average or variance
	            sum   = sum  + ddist * src(i, srcss, k, l)
	         ENDIF

 210	      CONTINUE

	      IF (trans.EQ.prgrd_good_pt ) THEN
	         dst(i, dstss, k, l) = sum
	      ELSEIF ( distance .EQ. 0.0 ) THEN
	         dst(i, dstss, k, l) = bad_dst
	      ELSEIF (trans.EQ.prgrd_sum ) THEN
	         dst(i, dstss, k, l) = sum
	      ELSE
* ... divide by distance to get average
	         dst(i, dstss, k, l) = sum / distance
	      ENDIF

 260	   CONTINUE
 290	   CONTINUE

* along Z axis
        ELSEIF ( axis .EQ. z_dim ) THEN

           DO 390 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO 390 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
           DO 390 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* loop through the each destination grid box
	   DO 360 dstss = dstss_lo,dstss_hi
	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

*  check if requested region is out of source range
              IF ( srcss_lo .EQ. unspecified_int4
     .	     .OR.  srcss_hi .EQ. unspecified_int4 ) THEN
	         dst(i,j,dstss,l) = bad_dst
	         goto 360
	      ENDIF

	      xdst_lo = TM_WORLD(dstss, dst_grid, axis, box_lo_lim )
	      xdst_hi = TM_WORLD(dstss, dst_grid, axis, box_hi_lim )

* add all partial or complete boxes of data from the source grid
	      sum = 0.0D0
	      distance = 0.0D0
	      DO 310 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src( i, j, srcss, l ) .EQ. bad_src ) GOTO 310

	         xsrc_min = TM_WORLD(srcss,src_grid,axis,box_lo_lim)
	         xsrc_max = TM_WORLD(srcss,src_grid,axis,box_hi_lim)

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi )

	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box
	         distance  = distance + ddist
	         IF (trans .EQ. prgrd_sum ) THEN
	            sum   = sum  + src(i, j, srcss, l)
     .			         * ddist/(xsrc_max-xsrc_min)
	         ELSEIF (trans .EQ. prgrd_good_pt ) THEN
	            sum   = sum  + ddist/(xsrc_max-xsrc_min)

	         ELSE	! average or variance
	            sum   = sum  + ddist * src(i, j, srcss, l)
	         ENDIF

 310	      CONTINUE

	      IF ( trans.EQ.prgrd_good_pt ) THEN
	         dst(i, j, dstss, l) = sum
	      ELSEIF ( distance .EQ. 0.0 ) THEN
	         dst(i, j, dstss, l) = bad_dst
	      ELSEIF (trans.EQ.prgrd_sum ) THEN
	         dst(i, j, dstss, l) = sum
	      ELSE
* ... divide by distance to get average
	         dst(i, j, dstss, l) = sum / distance
	      ENDIF

 360	   CONTINUE
 390	   CONTINUE

* along T axis
        ELSEIF ( axis .EQ. T_dim ) THEN

           DO 490 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO 490 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
           DO 490 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* loop through the each destination grid box
	   DO 460 dstss = dstss_lo,dstss_hi
	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

*  check if requested region is out of source range
              IF ( srcss_lo .EQ. unspecified_int4
     .	     .OR.  srcss_hi .EQ. unspecified_int4 ) THEN
	         dst(i,j,k,dstss) = bad_dst
	         goto 460
	      ENDIF

* USE SOURCE GRID TIME/DATE ENCODINGS FOR ALL CALCULATIONS 6/94
* initialization of TDEST_WORLD is done from GET_AVE_LIMS

	      xdst_lo = TDEST_WORLD(dstss, dst_grid, axis, box_lo_lim )
	      xdst_hi = TDEST_WORLD(dstss, dst_grid, axis, box_hi_lim )

* add all partial or complete boxes of data from the source grid
	      sum = 0.0D0
	      distance = 0.0D0
	      DO 410 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src( i, j, k, srcss ) .EQ. bad_src ) GOTO 410

	         xsrc_min = TM_WORLD(srcss,src_grid,axis,box_lo_lim)
	         xsrc_max = TM_WORLD(srcss,src_grid,axis,box_hi_lim)

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi )

	         ddist   = xsrc_hi - xsrc_lo

* ... contribution from this source box
	         distance  = distance + ddist
	         IF (trans .EQ. prgrd_sum ) THEN
	            sum   = sum  + src(i, j, k, srcss)
     .			         * ddist/(xsrc_max-xsrc_min)
	         ELSEIF (trans .EQ. prgrd_good_pt ) THEN
	            sum   = sum  + ddist/(xsrc_max-xsrc_min)

	         ELSE	! average or variance
	            sum   = sum  + ddist * src(i, j, k, srcss)
	         ENDIF
 410	      CONTINUE


	      IF ( trans.EQ.prgrd_good_pt ) THEN
	         dst(i, j, k, dstss) = sum
	      ELSEIF ( distance .EQ. 0.0 ) THEN
	         dst(i, j, k, dstss) = bad_dst
	      ELSEIF (trans.EQ.prgrd_sum ) THEN
	         dst(i, j, k, dstss) = sum
	      ELSE
* ... divide by distance to get average
	         dst(i, j, k, dstss) = sum / distance
	      ENDIF

 460	   CONTINUE
 490	   CONTINUE

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
           DO 590 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO 590 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO 590 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)

* loop through the each destination grid box
	   DO 560 dstss = dstss_lo,dstss_hi

	      IF ( dst(dstss, j, k, l) .EQ. bad_dst ) GOTO 560

	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

	      xdst_lo = TM_WORLD(dstss, dst_grid, axis, box_lo_lim )
	      xdst_hi = TM_WORLD(dstss, dst_grid, axis, box_hi_lim )

* add all partial or complete boxes of data from the source grid
	      sum = 0.0D0
	      distance = 0.0D0
	      DO 510 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src( srcss, j, k, l ) .EQ. bad_src ) GOTO 510

	         xsrc_lo = MAX( TM_WORLD(srcss,src_grid,axis,box_lo_lim),
     .                          xdst_lo )
	         xsrc_hi = MIN( TM_WORLD(srcss,src_grid,axis,box_hi_lim),
     .                          xdst_hi )
	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box
	         distance  = distance + ddist
	         dev = DBLE(src(srcss, j, k, l)) - DBLE(dst(dstss, j, k, l))
	         sum   = sum  + ddist * dev*dev

 510	      CONTINUE

* ... divide by weight to get variance
	      dst(dstss, j, k, l) = sum / distance

 560	   CONTINUE
 590	   CONTINUE

* along Y axis
        ELSEIF ( axis .EQ. Y_dim ) THEN

           DO 690 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO 690 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO 690 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* loop through the each destination grid box
	   DO 660 dstss = dstss_lo,dstss_hi

	      IF ( dst(i, dstss, k, l) .EQ. bad_dst ) GOTO 660

	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

	      xdst_lo = TM_WORLD(dstss, dst_grid, axis, box_lo_lim )
	      xdst_hi = TM_WORLD(dstss, dst_grid, axis, box_hi_lim )

* add all partial or complete boxes of data from the source grid
	      sum = 0.0D0
	      distance = 0.0D0
	      DO 610 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src( i, srcss, k, l ) .EQ. bad_src ) GOTO 610

	         xsrc_lo = MAX( TM_WORLD(srcss,src_grid,axis,box_lo_lim),
     .                          xdst_lo )
	         xsrc_hi = MIN( TM_WORLD(srcss,src_grid,axis,box_hi_lim),
     .                          xdst_hi )
	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box
	         distance  = distance + ddist
	         dev = DBLE(src(i, srcss, k, l)) - DBLE(dst(i, dstss, k, l))
	         sum   = sum  + ddist * dev*dev

 610	      CONTINUE

* ... divide by weight to get variance
	      dst(i, dstss, k, l) = sum / distance

 660	   CONTINUE
 690	   CONTINUE

* along Z axis
        ELSEIF ( axis .EQ. z_dim ) THEN

           DO 790 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO 790 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
           DO 790 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* loop through the each destination grid box
	   DO 760 dstss = dstss_lo,dstss_hi

	      IF ( dst(i, j, dstss, l) .EQ. bad_dst ) GOTO 760

	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

	      xdst_lo = TM_WORLD(dstss, dst_grid, axis, box_lo_lim )
	      xdst_hi = TM_WORLD(dstss, dst_grid, axis, box_hi_lim )

* add all partial or complete boxes of data from the source grid
	      sum = 0.0D0
	      distance = 0.0D0
	      DO 710 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src( i, j, srcss, l ) .EQ. bad_src ) GOTO 710

	         xsrc_lo = MAX( TM_WORLD(srcss,src_grid,axis,box_lo_lim),
     .                          xdst_lo )
	         xsrc_hi = MIN( TM_WORLD(srcss,src_grid,axis,box_hi_lim),
     .                          xdst_hi )
	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box
	         distance  = distance + ddist
	         dev = DBLE(src(i, j, srcss, l)) - DBLE(dst(i, j, dstss, l))
	         sum   = sum  + ddist * dev*dev

 710	      CONTINUE

* ... divide by weight to get variance
	      dst(i, j, dstss, l) = sum / distance

 760	   CONTINUE
 790	   CONTINUE

* along T axis
        ELSEIF ( axis .EQ. T_dim ) THEN

           DO 890 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO 890 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
           DO 890 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* loop through the each destination grid box
	   DO 860 dstss = dstss_lo,dstss_hi

	      IF ( dst(i, j, k, dstss) .EQ. bad_dst ) GOTO 860

	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

* USE SOURCE GRID TIME/DATE ENCODINGS FOR ALL CALCULATIONS 6/94
* initialization of TDEST_WORLD is done from GET_AVE_LIMS

	      xdst_lo = TDEST_WORLD(dstss, dst_grid, axis, box_lo_lim )
	      xdst_hi = TDEST_WORLD(dstss, dst_grid, axis, box_hi_lim )

* add all partial or complete boxes of data from the source grid
	      sum = 0.0D0
	      distance = 0.0D0
	      DO 810 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src( i, j, k, srcss ) .EQ. bad_src ) GOTO 810

	         xsrc_lo = MAX( TM_WORLD(srcss,src_grid,axis,box_lo_lim),
     .                          xdst_lo )
	         xsrc_hi = MIN( TM_WORLD(srcss,src_grid,axis,box_hi_lim),
     .                          xdst_hi )
	         ddist   = xsrc_hi - xsrc_lo

* ... contribution from this source box
	         distance  = distance + ddist
	         dev = DBLE(src(i, j, k, srcss)) - DBLE(dst(i, j, k, dstss))
	         sum   = sum  + ddist * dev*dev
 810	      CONTINUE

* ... divide by weight to get variance
	      dst(i, j, k, dstss) = sum / distance

 860	   CONTINUE
 890	   CONTINUE

        ENDIF

	RETURN
	END
