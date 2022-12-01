	SUBROUTINE DO_WGT(	action, idim, com_lo, com_hi,
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx,
     .				box )

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
* Return the weights as they would be used for an integral
* on a single axis, with correction for latitude if needed.
*
* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA 
* 7/2014
*
* V692 7/14 *acm* new transform @WGT, ticket 2180
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

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
	INTEGER i, j, k, l, m, n, grid, lo, hi, axis
	REAL	BOX_SIZE, unit, bad_com, bad_res, cos_factor,
     .          ybxlo, ybxhi, yh
	REAL*8	TM_WORLD, dsum, bsum

* calling argument declarations:
	INTEGER	action, idim, com_lo, com_hi, com_mr, com_cx, res_mr, res_cx
	REAL    com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
     .		box( com_lo:com_hi )

* --- end of introductory code ---

* initialize
	grid = mr_grid( com_mr )
	axis = grid_line( idim, grid )
	lo   = com_lo
	hi   = com_hi
	bad_com = mr_bad_data( com_mr )
	bad_res = mr_bad_data( res_mr )
	integrate = action .EQ. isact_integ_def
        point = .FALSE.

* get and save grid box sizes (weights) just once for efficiency
* (end boxes may be only partially weighted)
        IF ( lo .EQ. hi ) THEN    ! 8/92
           box(lo) = cx_hi_ww(idim,res_cx) - cx_lo_ww(idim,res_cx)
           point = box(lo) .EQ. 0.0
        ELSE
	   box(lo) = TM_WORLD(lo, grid, idim, box_hi_lim)
     .		   - cx_lo_ww(idim,res_cx)
	   DO i = lo+1, hi-1
	      box(i)  = BOX_SIZE( i, grid, idim )
 	   ENDDO
	   box(hi) = cx_hi_ww(idim,res_cx)
     .		   - TM_WORLD(hi, grid, idim, box_lo_lim)
        ENDIF

* perform integrations on standard units if possible
	
	unit = un_convert( line_unit_code(axis) )
	IF (ITSA_TRUEMONTH_AXIS(axis)) unit = un_convert(pun_day)
	IF (unit .NE. 1.) THEN
	   DO i = lo, hi
	      box(i) = box(i) * unit
 	   ENDDO
        ENDIF

* Compute weights
* ALONG X AXIS
* (note: X is a special case because it requires COSINE(lat) )
	IF ( idim .EQ. x_dim ) THEN
	   cos_factor = 1.
	   need_cos = GEOG_COS_FACTOR( idim, grid )
     .                .AND. cx_lo_s2(com_cx).NE.unspecified_int4
	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	      IF ( need_cos ) THEN
	         cos_factor = COS( deg2rad *       
     .             TM_WORLD(j, grid, y_dim, box_middle) ) 
                 ybxlo = deg2rad* 
     .             TM_WORLD(j,grid,y_dim,box_lo_lim)
                 ybxhi = deg2rad* 
     .             TM_WORLD(j,grid,y_dim,box_hi_lim)
                 yh = ABS(ybxhi - ybxlo)
	         cos_factor = ABS( SIN(ybxhi) - SIN(ybxlo) )/ yh
 
	         cos_factor = COS( deg2rad*
     .               TM_WORLD(j, grid, y_dim, box_middle) )* 
     .               ABS(SIN(yh/2)/(yh/2) )
	      ENDIF
	   
	   DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	      IF ( point ) THEN  
	         res(i,j,k,l,m,n) = 0.
	      ELSE
	         DO i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	          res(i,j,k,l,m,n) = box(i)* cos_factor
	         ENDDO
	      ENDIF

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	   ENDDO


* ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	      IF ( point ) THEN
	         res(i,j,k,l,m,n) = 0.
	      ELSE
	         DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	            res(i,j,k,l,m,n) = box(j)
		 ENDDO
	      ENDIF

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              IF ( point ) THEN 
	         res(i,j,k,l,m,n) = 0.
	      ELSE
	         DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	            res(i,j,k,l,m,n) = box(k)
		 ENDDO
              ENDIF

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* ALONG T AXIS
	ELSEIF ( idim .EQ. t_dim ) THEN
	   DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              IF ( point ) THEN
	      res(i,j,k,l,m,n) = 0.
	   ELSE
	      DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	         res(i,j,k,l,m,n) = box(l)
	      ENDDO   
	   ENDIF

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* ALONG E AXIS
	ELSEIF ( idim .EQ. e_dim ) THEN
	   DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              IF ( point ) THEN
	         res(i,j,k,l,m,n) = 0.
	      ELSE
	      DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	         res(i,j,k,l,m,n) = box(m)
	      ENDDO
           ENDIF

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO


* ALONG F AXIS

	ELSEIF ( idim .EQ. f_dim ) THEN
	   DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              IF ( point ) THEN
	      res(i,j,k,l,m,n) = 0.
	   ELSE
	      DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	         res(i,j,k,l,m,n) = box(n)
	      ENDDO
	   ENDIF

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	ENDIF

	RETURN
	END
