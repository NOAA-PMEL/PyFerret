	SUBROUTINE IIN_REGRID_1_AXIS( cx_lims, idim, src, msrc, dst, mdst, 
     .				 ss21, wksize, status )

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
* grid dst_grid by computing the integral over boxes along one axis.
* msrc and mdst point to data structures in COMMON/XVARIABLES/ which
* specify the subscript bounds of src and dst within their 
* respective grids.
* ss21 holds the indices of source grid boxes (on src_grid) for box
* limits on dst_grid.
*
* programmer - Ansley Manke, from ave_regrid_1_axis.F
* 	NOAA/PMEL,Seattle,WA - Tropical Modeling and Analysis Program
*       2/2018

* See ticket 2346/ issue 1618: we are mimicking these Ferret commands:
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
* V741  5/18 *acm* issue 1868: gridding transforms that will grid to same axis
*                  The @IIN regridding is only allowed to source axis or to an
*                  axis whose coordinates are the cell edges of the surce

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xcontext.cmn'
	include 'xmem_subsc.cmn'
	include 'xvariables.cmn'
	include 'xunits.cmn_text'
	external xunits_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data

* calling argument declarations:
	INTEGER cx_lims, idim, trans, msrc, mdst, wksize, status
	INTEGER ss21(wksize)

	REAL    src( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .	             m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .	        dst( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .	             m2lot:m2hit,m2loe:m2hie,m2lof:m2hif )

* local variable declarations:

        LOGICAL         ITSA_TRUEMONTH_AXIS,GEOG_COS_FACTOR, TM_FPEQ,
     .			need_cos, ok_neg111, bad, ioffset, offset
        INTEGER		dstss,srcss,src_grid,dst_grid,i,j,k,l,m,n
	INTEGER	     	srcss_lo,srcss_hi, dstss_lo,dstss_hi, idst

	REAL		ddist, bad_src, bad_dst, unit
	REAL*8		TM_WORLD, TDEST_WORLD, sum,
     .			xsrc_min, xsrc_max, xsrc_lo, xsrc_hi, xsrc_mid, 
     .			xdst_lo, xdst_hi, xdst_mid, first, 
     .			bx, ybxlo, ybxhi, cos_factor
******************************************************************************

* initialize
	status = ferr_ok
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

* Checks: this computation only on specific destination axis.
* Same units?
        bad = unit .NE. un_convert( line_unit_code(grid_line(idim,dst_grid)) )
	IF (bad) GOTO 5100

* Same box sizes?

	idst = dstss_lo
	xdst_lo = TM_WORLD(idst, dst_grid, idim, box_lo_lim )
	xdst_hi = TM_WORLD(idst, dst_grid, idim, box_hi_lim )
	xdst_mid = TM_WORLD(idst, dst_grid, idim, box_middle )

* Check that the regions match and adjust subscript ranges. When
* regriddding to/from a sub-range on larger axes, the context sent 
* in will have an extra grid cell on the source axis

	xsrc_lo = TM_WORLD(srcss_lo, src_grid, idim, box_lo_lim)
	xsrc_hi = TM_WORLD(srcss_lo, src_grid, idim, box_hi_lim)

	IF (TM_FPEQ(xsrc_hi,xdst_lo)) THEN
	   srcss_lo = srcss_lo + 1
	   xdst_hi = TM_WORLD(dstss_hi, dst_grid, idim, box_hi_lim )
	   xsrc_lo = TM_WORLD(srcss_hi, src_grid, idim, box_lo_lim)
	   IF (TM_FPEQ(xsrc_lo,xdst_hi)) srcss_hi = srcss_hi - 1
	   xdst_hi = TM_WORLD(idst, dst_grid, idim, box_hi_lim )
	ENDIF
	
	IF (TM_FPEQ(xsrc_hi,xdst_mid)) THEN
	   srcss_lo = srcss_lo + 1
	   xdst_mid = TM_WORLD(dstss_hi, dst_grid, idim, box_middle )
	   xsrc_lo = TM_WORLD(srcss_hi, src_grid, idim, box_lo_lim)
	   IF (TM_FPEQ(xsrc_lo,xdst_mid)) srcss_hi = srcss_hi - 1
	   xdst_mid = TM_WORLD(idst, dst_grid, idim, box_middle )
	ENDIF

* See if the regridding is to the same axis or to an axis defined as the 
* cell edges of the source axis (offset=true).

	offset = .FALSE.

	DO i = srcss_lo, srcss_hi

	   xsrc_lo = TM_WORLD(i, src_grid, idim, box_lo_lim)
	   xsrc_hi = TM_WORLD(i, src_grid, idim, box_hi_lim)

	   IF (xsrc_lo .GT. xdst_lo) THEN

* The destination grid is the edges of the source?

	      ioffset = TM_FPEQ(xsrc_lo,xdst_mid)
	      IF (i.EQ.srcss_lo) THEN
	         offset = ioffset  
	      ELSE
	         IF (offset .AND. .NOT.ioffset) GOTO 5100
		 offset = ioffset .AND. TM_FPEQ(xsrc_lo,xdst_mid) 
	      ENDIF

* same axis for src and dst? check the box-middle as well as edges

	   ELSEIF (TM_FPEQ(xsrc_lo,xdst_lo)) THEN
	      xsrc_mid = TM_WORLD(i, src_grid, idim, box_middle)
	      IF ( .NOT. TM_FPEQ(xdst_mid, xsrc_mid) ) GOTO 5100

* Not same axis, refuse to return a result
	   ELSE 
	      GOTO 5100
	   ENDIF

	   idst = idst + 1
	   xdst_lo = TM_WORLD(idst, dst_grid, idim, box_lo_lim )
	   xdst_hi = TM_WORLD(idst, dst_grid, idim, box_hi_lim )
	   xdst_mid = TM_WORLD(idst, dst_grid, idim, box_middle )

	ENDDO 

	IF (ITSA_TRUEMONTH_AXIS(idim)) unit = un_convert(pun_day)

* determine the source grid boxes containing the destination box edges
	CALL GET_AVE_LIMS( srcss_lo, srcss_hi, src_grid,
     .			   dstss_lo, dstss_hi, dst_grid,
     .			   idim, ss21 )

* along X axis
        IF ( idim .EQ. x_dim ) THEN
	
	   cos_factor = 1.
	   need_cos = GEOG_COS_FACTOR( idim, src_grid )
     .                .AND. cx_lo_s2(cx_lims).NE.unspecified_int4

           DO 190 n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
           DO 190 m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
           DO 190 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
           DO 190 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
           DO 190 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)

	   IF ( need_cos ) THEN
              ybxlo = deg2rad* 
     .            TM_WORLD(j,src_grid,y_dim,box_lo_lim)
              ybxhi = deg2rad* 
     .            TM_WORLD(j,src_grid,y_dim,box_hi_lim)
              bx = ABS(ybxhi - ybxlo)
	      cos_factor = ABS( SIN(ybxhi) - SIN(ybxlo) )/ bx
	   ENDIF 

	   sum = 0.0D0
	   dst(dstss_lo, j, k, l, m, n) = 0.D0

	   IF (ss21(1)  .EQ. unspecified_int4) 
     .	     dst(dstss_lo, j, k, l, m, n) = bad_dst

* loop through the each destination grid box, add the source box-size*value
*  to the running sum
	   DO 160 dstss = dstss_lo, dstss_hi
	      srcss_lo = ss21(dstss-dstss_lo+1)
	      srcss_hi = ss21(dstss-dstss_lo+2)

*  check if requested region is out of source range
              IF ( (srcss_lo .EQ. unspecified_int4
     .	     .OR.  srcss_hi .EQ. unspecified_int4)
     .       .AND. .NOT.(ok_neg111) ) THEN
	         dst(dstss,j,k,l,m,n) = bad_dst
	         goto 160
	      ENDIF

	      xdst_lo = unit* TM_WORLD(dstss-1, dst_grid, idim, box_lo_lim )
	      xdst_hi = unit* TM_WORLD(dstss, dst_grid, idim, box_hi_lim )

* add boxes of data from the source grid

	      srcss = srcss_lo

* ... ignore missing source data
	      IF ( src(srcss, j, k, l, m, n) .EQ. bad_src ) GOTO 110

	      xsrc_lo = unit* TM_WORLD(srcss,src_grid,idim,box_lo_lim)
	      xsrc_hi = unit* TM_WORLD(srcss,src_grid,idim,box_hi_lim)

	      IF (xsrc_hi.LT.xdst_lo .OR. xsrc_lo.GT.xdst_hi)  GOTO 110

	      ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box add to running sum
	      sum   = sum  + cos_factor* ddist * src(srcss, j, k, l, m, n)

 110	      CONTINUE

* ... Start the result at 0.
	      IF (dstss .EQ. 1) first = sum
	      IF (offset) THEN
	         dst(dstss, j, k, l, m, n) = sum - first
	      ELSE
	         IF (dstss.LT.dstss_hi) dst(dstss+1, j, k, l, m, n) = sum
	      ENDIF

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

* loop through the each destination grid box, add the source box-size*value
*  to the running sum
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

	      xdst_lo = unit* TM_WORLD(dstss-1, dst_grid, idim, box_lo_lim )
	      xdst_hi = unit* TM_WORLD(dstss, dst_grid, idim, box_hi_lim )

* add all complete boxes of data from the source grid

	      srcss = srcss_lo

* ... ignore missing source data
	      IF ( src(i, srcss, k, l, m, n) .EQ. bad_src ) GOTO 210

	      xsrc_lo = unit* TM_WORLD(srcss,src_grid,idim,box_lo_lim)
	      xsrc_hi = unit* TM_WORLD(srcss,src_grid,idim,box_hi_lim)

	      IF (xsrc_hi.LT.xdst_lo .OR. xsrc_lo.GT.xdst_hi)  GOTO 210

	      ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box add to running sum
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

* loop through the each destination grid box, add the source box-size*value
*  to the running sum
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

	      xdst_lo = unit* TM_WORLD(dstss-1, dst_grid, idim, box_lo_lim )
	      xdst_hi = unit* TM_WORLD(dstss, dst_grid, idim, box_hi_lim )

* add all complete boxes of data from the source grid


	      srcss = srcss_lo

* ... ignore missing source data
	      IF ( src(i, j, srcss, l, m, n) .EQ. bad_src ) GOTO 310

	      xsrc_lo = unit* TM_WORLD(srcss,src_grid,idim,box_lo_lim)
	      xsrc_hi = unit* TM_WORLD(srcss,src_grid,idim,box_hi_lim)

	      IF (xsrc_hi.LT.xdst_lo .OR. xsrc_lo.GT.xdst_hi)  GOTO 310

	      ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box add to running sum
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
 
* loop through the each destination grid box, add the source box-size*value
*  to the running sum
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

	      xdst_lo = unit* TM_WORLD(dstss-1, dst_grid, idim, box_lo_lim )
	      xdst_hi = unit* TM_WORLD(dstss, dst_grid, idim, box_hi_lim )

* add all complete boxes of data from the source grid


	      srcss = srcss_lo
* ... ignore missing source data
	      IF ( src(i, j, k, srcss, m, n) .EQ. bad_src ) GOTO 410

	      xsrc_lo = unit* TM_WORLD(srcss,src_grid,idim,box_lo_lim)
	      xsrc_hi = unit* TM_WORLD(srcss,src_grid,idim,box_hi_lim)

	      IF (xsrc_hi.LT.xdst_lo .OR. xsrc_lo.GT.xdst_hi)  GOTO 410

	      ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box add to running sum
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

* loop through the each destination grid box, add the source box-size*value
*  to the running sum
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

	      xdst_lo = unit* TM_WORLD(dstss-1, dst_grid, idim, box_lo_lim )
	      xdst_hi = unit* TM_WORLD(dstss, dst_grid, idim, box_hi_lim )

* add all complete boxes of data from the source grid


	      srcss = srcss_lo
* ... ignore missing source data
	      IF ( src(i, j, k, l, srcss, n) .EQ. bad_src ) GOTO 510

	      xsrc_lo = unit* TM_WORLD(srcss,src_grid,idim,box_lo_lim)
	      xsrc_hi = unit* TM_WORLD(srcss,src_grid,idim,box_hi_lim)

	      IF (xsrc_hi.LT.xdst_lo .OR. xsrc_lo.GT.xdst_hi)  GOTO 510

	      ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box add to running sum
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

* loop through the each destination grid box, add the source box-size*value
*  to the running sum
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

* initialization of TDEST_WORLD is done from GET_AVE_LIMS

	      xdst_lo = unit* TM_WORLD(dstss-1, dst_grid, idim, box_lo_lim )
	      xdst_hi = unit* TM_WORLD(dstss, dst_grid, idim, box_hi_lim )

* add all complete boxes of data from the source grid

	      srcss = srcss_lo

* ... ignore missing source data
	      IF ( src(i, j, k, l, m, srcss) .EQ. bad_src ) GOTO 610

	      xsrc_lo = unit* TM_WORLD(srcss,src_grid,idim,box_lo_lim)
	      xsrc_hi = unit* TM_WORLD(srcss,src_grid,idim,box_hi_lim)

	      IF (xsrc_hi.LT.xdst_lo .OR. xsrc_lo.GT.xdst_hi)  GOTO 610

	      ddist   = xsrc_hi-xsrc_lo

* ... contribution from this source box add to running sum

                 sum   = sum  + ddist * src(i, j, k, l, m, srcss)
 610	      CONTINUE

	      dst(i, j, k, l, m, dstss) = sum

 660	   CONTINUE
 690	   CONTINUE

        ENDIF

 5000	RETURN
 5100   status = ferr_invalid_command
	GOTO 5000
	END
