	SUBROUTINE DO_AVE_INT(	action, idim, com_lo, com_hi,
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx,
     .				box, weights )

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
* perform both accumulation of values needed for both averaging and integrating
* (definite) - since they differ only by a multiplying factor

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V200: ????
*      10/11/89 - modified array declarations using XMEM_SUBSC.CMN
* V230:  5/6/92 - COS correction checks that J location is defined
*       8/10/92 - weights were wrong for integration within a single box
*                 added "IF (point)" logic
* V68  *acm* 1/12  ifdef double_p for double-precision ferret.
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.
* V702 *sh* 11/16 this routine now performs accumulation only.  Initialization
*                 and finalizing called separately from IS_TRANS
* V72  *acm* 5/17 Fix ticket 2538: scatter/gather weights when average 
*                 over a "point"
* V7.6 *acm* 12/19 Issue 1964: Error when re-working this for memory in chunks.
*                 When setting box sizes the case of "IF ( lo .EQ. hi )" was 
*		  incorrect when a chunk is one box long. Gave too large a box 
*		  size, the length of the entire result axis, instead of the length
*                 for just this chunk.

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xunits.cmn_text'
	external xunits_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data

* internal variable declarations:
	LOGICAL	GEOG_COS_FACTOR, ITSA_TRUEMONTH_AXIS,
     .          need_cos, integrate, point
	INTEGER i, j, k, l, m, n, grid, pt, lo, hi, axis
	REAL	BOX_SIZE, unit, bad_com, bad_res, cos_factor,
     .          ybxlo, ybxhi, bx
	REAL*8	TM_WORLD, dsum, bsum

* calling argument declarations:
	INTEGER	action, idim, com_lo, com_hi, com_mr, com_cx, res_mr, res_cx
	REAL    com     ( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .                    m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          res     ( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .                    m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
     .          weights ( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .                    m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
     .		box( com_lo:com_hi )

* --- end of introductory code ---

* initialize
	grid = mr_grid( com_mr )
	axis = grid_line( idim, grid )
	pt   = cx_lo_ss( res_cx, idim )
	lo   = com_lo
	hi   = com_hi
	bad_com = mr_bad_data( com_mr )
	bad_res = mr_bad_data( res_mr )
	integrate = action .EQ. isact_integ_def
        point = .FALSE.

* get and save grid box sizes (weights) just once for efficiency
        
	DO i = lo, hi
	   box(i)  = BOX_SIZE( i, grid, idim )
	ENDDO
	IF ( lo .EQ. hi ) THEN
	   point = box(lo) .EQ. 0.0
	ELSE
* 11/16 ... end boxes may be only partial cells
	   IF (TM_WORLD(lo, grid, idim, box_lo_lim)
     .       .LT. cx_lo_ww(idim,res_cx)                ) box(lo) = 
     .			TM_WORLD(lo, grid, idim, box_hi_lim)
     .		      - cx_lo_ww(idim,res_cx)
           IF (TM_WORLD(hi, grid, idim, box_hi_lim)
     .       .GT. cx_hi_ww(idim,res_cx)                ) box(hi) =
     .			cx_hi_ww(idim,res_cx)
     .		      - TM_WORLD(hi, grid, idim, box_lo_lim)
	ENDIF

* perform integrations on standard units if possible
	IF ( integrate ) THEN
	   unit = un_convert( line_unit_code(axis) )
	   IF (ITSA_TRUEMONTH_AXIS(axis)) unit = un_convert(pun_day)
	   DO 110 i = lo, hi
 110	   box(i) = box(i) * unit
	ENDIF

* AVERAGE/INTEGRATE
* (is double precision summing necessary ? *sh* 6/89)
* ALONG X AXIS
* (note: X integration is a special case because it requires COSINE(lat) )
	IF ( idim .EQ. x_dim ) THEN
	   need_cos = GEOG_COS_FACTOR( idim, grid )
     .                .AND. cx_lo_s2(com_cx).NE.unspecified_int4
	   DO 210 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   cos_factor = 1.
	   IF ( need_cos .AND. integrate ) THEN
	      cos_factor = COS( deg2rad *       
     .             TM_WORLD(j, grid, y_dim, box_middle) ) 
              ybxlo = deg2rad* 
     .             TM_WORLD(j,grid,y_dim,box_lo_lim)
              ybxhi = deg2rad* 
     .             TM_WORLD(j,grid,y_dim,box_hi_lim)
              bx = ABS(ybxhi - ybxlo)
	      cos_factor = ABS( SIN(ybxhi) - SIN(ybxlo) )/ bx
	   ENDIF
	   DO 210 n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO 210 m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO 210 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 210 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
              IF ( point ) THEN  ! 8/92
                 IF ( com(lo,j,k,l,m,n) .EQ. bad_com ) THEN
                    res(pt,j,k,l,m,n) = bad_res
                 ELSEIF ( integrate ) THEN
                    res(pt,j,k,l,m,n) = 0.0
                 ELSE
                    res(pt,j,k,l,m,n) = com(lo,j,k,l,m,n)
                 ENDIF
		 weights(pt,j,k,l,m,n) = 1.0
                 GOTO 210
              ENDIF
	      dsum = 0.0D0	! data
	      bsum = 0.0D0	! boxes
	      DO 200 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) GOTO 200
	         bsum = bsum + box(i)
	         dsum = dsum + box(i)*com(i,j,k,l,m,n)
 200	      CONTINUE
	      res(pt,j,k,l,m,n) = res(pt,j,k,l,m,n) + dsum* cos_factor
	      weights(pt,j,k,l,m,n) = weights(pt,j,k,l,m,n) + bsum

 210	   CONTINUE

* ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO 310 n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO 310 m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO 310 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 310 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 310 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              IF ( point ) THEN   ! 8/92
                 IF ( com(i,lo,k,l,m,n) .EQ. bad_com ) THEN
                    res(i,pt,k,l,m,n) = bad_res
                 ELSEIF ( integrate ) THEN
                    res(i,pt,k,l,m,n) = 0.0
                 ELSE
                    res(i,pt,k,l,m,n) = com(i,lo,k,l,m,n)
                 ENDIF
		 weights(i,pt,k,l,m,n) = 1.0
                 GOTO 310
              ENDIF
	      dsum = 0.0D0	! data
	      bsum = 0.0D0	! boxes
	      DO 300 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) GOTO 300
	         bsum = bsum + box(j)
	         dsum = dsum + box(j)*com(i,j,k,l,m,n)
 300	      CONTINUE
	      res(i,pt,k,l,m,n) = res(i,pt,k,l,m,n) + dsum
	      weights(i,pt,k,l,m,n) = weights(i,pt,k,l,m,n) + bsum
 310	   CONTINUE

* ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 410 n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO 410 m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO 410 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 410 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 410 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              IF ( point ) THEN    ! 8/92
                 IF ( com(i,j,lo,l,m,n) .EQ. bad_com ) THEN
                    res(i,j,pt,l,m,n) = bad_res
                 ELSEIF ( integrate ) THEN
                    res(i,j,pt,l,m,n) = 0.0
                 ELSE
                    res(i,j,pt,l,m,n) = com(i,j,lo,l,m,n)
                 ENDIF
                 weights(i,j,pt,l,m,n) = 1.0
                 GOTO 410
	      ENDIF
	      dsum = 0.0D0	! data
	      bsum = 0.0D0	! boxes
	      DO 400 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) GOTO 400
	         bsum = bsum + box(k)
	         dsum = dsum + box(k)*com(i,j,k,l,m,n)
 400	      CONTINUE
              res(i,j,pt,l,m,n) = res(i,j,pt,l,m,n) + dsum
              weights(i,j,pt,l,m,n) = weights(i,j,pt,l,m,n) + bsum
 410	   CONTINUE

* ALONG T AXIS
	ELSEIF ( idim .EQ. t_dim ) THEN
	   DO 510 n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO 510 m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO 510 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 510 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 510 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              IF ( point ) THEN
                 IF ( com(i,j,k,lo,m,n) .EQ. bad_com ) THEN
                    res(i,j,k,pt,m,n) = bad_res
                 ELSEIF ( integrate ) THEN
                    res(i,j,k,pt,m,n) = 0.0
                 ELSE
                    res(i,j,k,pt,m,n) = com(i,j,k,lo,m,n)
                 ENDIF
		 weights(i,j,k,pt,m,n) = 1.0
                 GOTO 510
              ENDIF
	      dsum = 0.0D0	! data
	      bsum = 0.0D0	! boxes
	      DO 500 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) GOTO 500
	         bsum = bsum + box(l)
	         dsum = dsum + box(l)*com(i,j,k,l,m,n)
 500	      CONTINUE
* ... accumulate weighted values and weights
	      weights(i,j,k,pt,m,n) = weights(i,j,k,pt,m,n) + bsum
	      res    (i,j,k,pt,m,n) = res    (i,j,k,pt,m,n) + dsum
 510	   CONTINUE


* ALONG E AXIS
	ELSEIF ( idim .EQ. e_dim ) THEN
	   DO 610 n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO 610 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 610 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 610 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 610 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              IF ( point ) THEN
                 IF ( com(i,j,k,l,lo,n) .EQ. bad_com ) THEN
                    res(i,j,k,l,pt,n) = bad_res
                 ELSEIF ( integrate ) THEN
                    res(i,j,k,l,pt,n) = 0.0
                 ELSE
                    res(i,j,k,l,pt,n) = com(i,j,k,l,lo,n)
                 ENDIF
		 weights(i,j,k,l,pt,n) = 1.0
                 GOTO 610
              ENDIF
	      dsum = 0.0D0	! data
	      bsum = 0.0D0	! boxes
	      DO 600 m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) GOTO 600
	         bsum = bsum + box(m)
	         dsum = dsum + box(m)*com(i,j,k,l,m,n)
 600	      CONTINUE
	      res(i,j,k,l,pt,n) = res(i,j,k,l,pt,n) + dsum
	      weights(i,j,k,l,pt,n) = weights(i,j,k,l,pt,n) + bsum
 610	   CONTINUE


* ALONG F AXIS
	ELSEIF ( idim .EQ. f_dim ) THEN
	   DO 710 m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO 710 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 710 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 710 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 710 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              IF ( point ) THEN
                 IF ( com(i,j,k,l,m,lo) .EQ. bad_com ) THEN
                    res(i,j,k,l,m,pt) = bad_res
                 ELSEIF ( integrate ) THEN
                    res(i,j,k,l,m,pt) = 0.0
                 ELSE
                    res(i,j,k,l,m,pt) = com(i,j,k,l,m,lo)
                 ENDIF
		 weights(i,j,k,l,m,pt) = 1.0
                 GOTO 710
              ENDIF
	      dsum = 0.0D0	! data
	      bsum = 0.0D0	! boxes
	      DO 700 n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) GOTO 700
	         bsum = bsum + box(n)
	         dsum = dsum + box(n)*com(i,j,k,l,m,n)
 700	      CONTINUE
	      res(i,j,k,l,m,pt) = res(i,j,k,l,m,pt) + dsum
	      weights(i,j,k,l,m,pt) = weights(i,j,k,l,m,pt) + bsum
 710	   CONTINUE

	ENDIF

	RETURN
	END
