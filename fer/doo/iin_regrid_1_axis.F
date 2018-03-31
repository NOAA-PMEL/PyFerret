	SUBROUTINE IIN_REGRID_1_AXIS( cx_lims, idim,
     .				 dst_lox, dst_hix, dst_loy, dst_hiy,
     .				 src, msrc, dst, mdst, ss21, wksize )

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
* Using data values from src organized on grid src_grid, regrid to
* grid dst_grid by computing the integral over boxes along one axis
* msrc and mdst point to data structures in COMMON/XVARIABLES/ which
* specify the subscript bounds of src and dst within their 
* respective grids.
* ss21 holds the indices of source grid boxes (on src_grid) for box
* limits on dst_grid.
*
* programmer - Ansley Manke, from ave_regrid_1_axis.F
* 	NOAA/PMEL,Seattle,WA - Tropical Modeling and Analysis Program
*       2/2018

* See ticket 2346/ issue 1618: we are mimicing these Ferret commands:
* Define the indefinite integral, but locate the result at the top of 
* the grid cells instead of the grid cell centers.
*
*  let xlo = xboxlo[gx=a]
*  def ax/x xax_edges = xcat(xlo[i=1], xboxhi[gx=a])
*  let a_iin_edges = a_iin[gx=xax_edges@asn]
*  let a_int = IF i[gx=xax_edges] EQ 1 THEN 0 ELSE a_iin_edges[x=@shf:-1]
*
* Assumes the destination grid is be the cell edges of the source grid.
* TODO: Is this really required? To check this, require that 
*       xdst_lo=xsrc_min, xdst_hi=xsrc_max 
*
* 3/30/18 *acm* Return results in world-coordinate units


        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xcontext.cmn'
	include 'xmem_subsc.cmn'
	include 'xvariables.cmn'
	include 'xunits.cmn_text'
	external xunits_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data

* calling argument declarations:
	INTEGER cx_lims, idim, trans, msrc, mdst,
     .          dst_lox, dst_hix, dst_loy, dst_hiy , wksize
	INTEGER ss21(wksize)

	REAL    src( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .	             m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .	        dst( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .	             m2lot:m2hit,m2loe:m2hie,m2lof:m2hif )

* local variable declarations:

        LOGICAL         ITSA_TRUEMONTH_AXIS, ok_neg111
        INTEGER		dstss,srcss,src_grid,dst_grid,i,j,k,l,m,n
	INTEGER	     	srcss_lo,srcss_hi, dstss_lo,dstss_hi

	REAL		ddist, bad_src, bad_dst, unit
	REAL*8		TM_WORLD, TDEST_WORLD, sum,
     .			xsrc_min, xsrc_max, xsrc_lo, xsrc_hi,
     .			xdst_lo, xdst_hi
******************************************************************************

* initialize
	src_grid = mr_grid( msrc )
	dst_grid = mr_grid( mdst )

* limits for calculation
        srcss_lo = mr_lo_ss(msrc,idim)
        srcss_hi = mr_hi_ss(msrc,idim)

* Range of subscripts might be negative, and contain the value of unspecified_int4 
* among them.  ok_neg111 distinguishes between this and the flag unspecified_int4.

        ok_neg111 = srcss_lo .LT. unspecified_int4 .AND.
     .              srcss_hi .GT. unspecified_int4

        dstss_lo = cx_lo_ss(cx_lims,idim)
        dstss_hi = cx_hi_ss(cx_lims,idim)

* flag for bad/missing data
        bad_src = mr_bad_data(msrc)
        bad_dst = mr_bad_data(mdst)


* perform integrations on standard units if possible
	unit = un_convert( line_unit_code(grid_line(idim,src_grid)) )
	IF (ITSA_TRUEMONTH_AXIS(idim)) unit = un_convert(pun_day)

* determine the source grid boxes containing the destination box edges
	CALL GET_AVE_LIMS( srcss_lo, srcss_hi, src_grid,
     .			   dstss_lo, dstss_hi, dst_grid,
     .			   idim, ss21 )

* along X axis
        IF ( idim .EQ. x_dim ) THEN
           DO 190 n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
           DO 190 m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
           DO 190 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO 190 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO 190 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)

	   sum = 0.0D0
	   dst(dstss_lo, j, k, l, m, n) = 0.D0
	   IF (ss21(1)  .EQ. unspecified_int4) 
     .	     dst(dstss_lo, j, k, l, m, n) = bad_dst

* loop through the each destination grid box
	   DO 160 dstss = dstss_lo+1, dstss_hi
	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

*  check if requested region is out of source range
              IF ( (srcss_lo .EQ. unspecified_int4
     .	     .OR.  srcss_hi .EQ. unspecified_int4)
     .       .AND. .NOT.(ok_neg111) ) THEN
	         dst(dstss,j,k,l,m,n) = bad_dst
	         goto 160
	      ENDIF

	      xdst_lo = unit* TM_WORLD(dstss-1, dst_grid, idim, box_middle )
	      xdst_hi = unit* TM_WORLD(dstss, dst_grid, idim, box_middle )

* add all partial or complete boxes of data from the source grid

* IF source box completely encloses the dest box, then srcss_lo=srcss_hi
* and we can just stick the source point into the destination (for avg or var)
	      DO 110 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(srcss, j, k, l, m, n) .EQ. bad_src ) GOTO 110

	         xsrc_min = unit* TM_WORLD(srcss,src_grid,idim,box_lo_lim)
	         xsrc_max = unit* TM_WORLD(srcss,src_grid,idim,box_hi_lim)

	         IF (xsrc_max.LT.xdst_lo .OR. xsrc_min.GT.xdst_hi) CYCLE

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi )

	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box add to running sum
                 sum   = sum  + ddist * src(srcss, j, k, l, m, n)

 110	      CONTINUE

	      dst(dstss, j, k, l, m, n) = sum


 160	   CONTINUE
 190	   CONTINUE

* along Y axis
        ELSEIF ( idim .EQ. Y_dim ) THEN

	   DO 290 n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
           DO 290 m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
           DO 290 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO 290 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO 290 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

           sum = 0.0D0
	   dst(i, dstss_lo, k, l, m, n) = 0.D0
	   IF (ss21(1)  .EQ. unspecified_int4) 
     .	     dst(i, dstss_lo, k, l, m, n) = bad_dst

* loop through the each destination grid box
	   DO 260 dstss = dstss_lo+1, dstss_hi
	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

*  check if requested region is out of source range
              IF ( (srcss_lo .EQ. unspecified_int4
     .	     .OR.  srcss_hi .EQ. unspecified_int4)
     .       .AND. .NOT.(ok_neg111) ) THEN
	         dst(i,dstss,k,l,m,n) = bad_dst
	         goto 260
	      ENDIF

	      xdst_lo = unit* TM_WORLD(dstss-1, dst_grid, idim, box_middle )
	      xdst_hi = unit* TM_WORLD(dstss, dst_grid, idim, box_middle )

* add all partial or complete boxes of data from the source grid

* IF source box completely encloses the dest box, then srcss_lo=srcss_hi
* and we can just stick the source point into the destination (for avg or var)
	      DO 210 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, srcss, k, l, m, n) .EQ. bad_src ) GOTO 210

	         xsrc_min = unit* TM_WORLD(srcss,src_grid,idim,box_lo_lim)
	         xsrc_max = unit* TM_WORLD(srcss,src_grid,idim,box_hi_lim)

	         IF (xsrc_max.LT.xdst_lo .OR. xsrc_min.GT.xdst_hi) CYCLE

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi)

	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box
                 sum   = sum  + ddist * src(i, srcss, k, l, m, n)

 210	      CONTINUE

	      dst(i, dstss, k, l, m, n) = sum

 260	   CONTINUE
 290	   CONTINUE

* along Z axis
        ELSEIF ( idim .EQ. z_dim ) THEN

           DO 390 n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
           DO 390 m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
           DO 390 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO 390 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
           DO 390 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

	   sum = 0.0D0
	   dst(i, j, dstss_lo, l, m, n) = 0.D0
	   IF (ss21(1)  .EQ. unspecified_int4) 
     .	     dst(i, j, dstss_lo, l, m, n) = bad_dst

* loop through the each destination grid box
	   DO 360 dstss = dstss_lo+1, dstss_hi
	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

*  check if requested region is out of source range
              IF ( (srcss_lo .EQ. unspecified_int4
     .	     .OR.  srcss_hi .EQ. unspecified_int4)
     .       .AND. .NOT.(ok_neg111) ) THEN
	         dst(i,j,dstss,k,m,n) = bad_dst
	         goto 360
	      ENDIF

	      xdst_lo = unit* TM_WORLD(dstss-1, dst_grid, idim, box_middle )
	      xdst_hi = unit* TM_WORLD(dstss, dst_grid, idim, box_middle )

* add all partial or complete boxes of data from the source grid

* IF source box completely encloses the dest box, then srcss_lo=srcss_hi
* and we can just stick the source point into the destination (for avg or var)
	      DO 310 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, j, srcss, l, m, n) .EQ. bad_src ) GOTO 310

	         xsrc_min = unit* TM_WORLD(srcss,src_grid,idim,box_lo_lim)
	         xsrc_max = unit* TM_WORLD(srcss,src_grid,idim,box_hi_lim)

	         IF (xsrc_max.LT.xdst_lo .OR. xsrc_min.GT.xdst_hi) CYCLE

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi )

	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box
                 sum   = sum  + ddist * src(i, j, srcss, l, m, n)

 310	      CONTINUE

	      dst(i, j, dstss, l, m, n) = sum

 360	   CONTINUE
 390	   CONTINUE

* along T axis
        ELSEIF ( idim .EQ. T_dim ) THEN

           DO 490 n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
           DO 490 m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
           DO 490 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO 490 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
           DO 490 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

	   sum = 0.0D0
	   dst(i, j, k, dstss_lo, m, n) = 0.D0
	   IF (ss21(1)  .EQ. unspecified_int4) 
     .	     dst(i, j, k, dstss_lo, m, n) = bad_dst
 
* loop through the each destination grid box
	   DO 460 dstss = dstss_lo+1, dstss_hi
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

	      xdst_lo = unit* TM_WORLD(dstss-1, dst_grid, idim, box_middle )
	      xdst_hi = unit* TM_WORLD(dstss, dst_grid, idim, box_middle )

* add all partial or complete boxes of data from the source grid

* IF source box completely encloses the dest box, then srcss_lo=srcss_hi
* and we can just stick the source point into the destination (for avg or var)
	      DO 410 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, j, k, srcss, m, n) .EQ. bad_src ) GOTO 410

	         xsrc_min = unit* TM_WORLD(srcss,src_grid,idim,box_lo_lim)
	         xsrc_max = unit* TM_WORLD(srcss,src_grid,idim,box_hi_lim)

	         IF (xsrc_max.LT.xdst_lo .OR. xsrc_min.GT.xdst_hi) CYCLE

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi)

	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box
                 sum   = sum  + ddist * src(i, j, k, srcss, m, n)
		
 410	      CONTINUE

	      dst(i, j, k, dstss, m, n) = sum

 460	   CONTINUE
 490	   CONTINUE

* along E axis
        ELSEIF ( idim .EQ. e_dim ) THEN

           DO 590 n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
           DO 590 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO 590 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO 590 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
           DO 590 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

           sum = 0.0D0
	   dst(i, j, k, l, dstss_lo, n) = 0.D0
	   IF (ss21(1)  .EQ. unspecified_int4) 
     .	     dst(i, j, k, l, dstss_lo, n) = bad_dst

* loop through the each destination grid box
	   DO 560 dstss = dstss_lo+1, dstss_hi
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

	      xdst_lo = unit* TM_WORLD(dstss-1, dst_grid, idim, box_middle )
	      xdst_hi = unit* TM_WORLD(dstss, dst_grid, idim, box_middle )

* add all partial or complete boxes of data from the source grid

* IF source box completely encloses the dest box, then srcss_lo=srcss_hi
* and we can just stick the source point into the destination (for avg or var)
	      DO 510 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, j, k, l, srcss, n) .EQ. bad_src ) GOTO 510

	         xsrc_min = unit* TM_WORLD(srcss,src_grid,idim,box_lo_lim)
	         xsrc_max = unit* TM_WORLD(srcss,src_grid,idim,box_hi_lim)

	         IF (xsrc_max.LT.xdst_lo .OR. xsrc_min.GT.xdst_hi) CYCLE

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi)

	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box
                 sum   = sum  + ddist * src(i, j, k, l, srcss, n)
 510	      CONTINUE

	      dst(i, j, k, l, dstss, n) = sum

 560	   CONTINUE
 590	   CONTINUE

* along F axis
        ELSEIF ( idim .EQ. f_dim ) THEN

           DO 690 m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
           DO 690 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO 690 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO 690 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
           DO 690 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

           sum = 0.0D0
	   dst(i, j, k, l, m, dstss_lo) = 0.D0
	   IF (ss21(1)  .EQ. unspecified_int4) 
     .	     dst(i, j, k, l, m, dstss_lo) = bad_dst

* loop through the each destination grid box
	   DO 660 dstss = dstss_lo+1, dstss_hi
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

	      xdst_lo = unit* TM_WORLD(dstss-1, dst_grid, idim, box_middle )
	      xdst_hi = unit* TM_WORLD(dstss, dst_grid, idim, box_middle )

* add all partial or complete boxes of data from the source grid

* IF source box completely encloses the dest box, then srcss_lo=srcss_hi
* and we can just stick the source point into the destination (for avg or var)
	      DO 610 srcss = srcss_lo, srcss_hi

* ... ignore missing source data
	         IF ( src(i, j, k, l, m, srcss) .EQ. bad_src ) GOTO 610

	         xsrc_min = unit* TM_WORLD(srcss,src_grid,idim,box_lo_lim)
	         xsrc_max = unit* TM_WORLD(srcss,src_grid,idim,box_hi_lim)

	         IF (xsrc_max.LT.xdst_lo .OR. xsrc_min.GT.xdst_hi) CYCLE

	         xsrc_lo = MAX(xsrc_min,xdst_lo)
	         xsrc_hi = MIN(xsrc_max,xdst_hi)

	         ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box
                 sum   = sum  + ddist * src(i, j, k, l, m, srcss)
 610	      CONTINUE

	      dst(i, j, k, l, m, dstss) = sum

 660	   CONTINUE
 690	   CONTINUE

        ENDIF

	RETURN
	END
