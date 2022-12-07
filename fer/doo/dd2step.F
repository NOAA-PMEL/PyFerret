	SUBROUTINE DD2STEP ( idim, arg, com_lo, com_hi,
     .			     com, com_mr,
     .			     res, res_mr, res_cx,
     .			     box	)

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
* compute centered derivative of component along axis idim

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:	10/11/89 - modified array declarations using XMEM_SUBSC.CMN
* V230: 5/6/92 - COS correction checks that J location is defined         
* V68  *acm* 1/12  ifdef double_p for double-precision ferret.
*       2/12 *kms* Include E and F dimensions (use nferdims in tmap_dims.parm)
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include	'xprog_state.cmn'
	include 'xunits.cmn_text'
	external xunits_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data

* calling argument declarations:
	INTEGER	idim, com_lo, com_hi, com_mr, com_cx, res_mr, res_cx
	REAL	arg,
     .		box( com_lo:com_hi ),
     .          com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif )

* internal variable declarations:
	LOGICAL	GEOG_COS_FACTOR, ITSA_TRUEMONTH_AXIS, need_cos
	INTEGER i, j, k, l, m, n, ii, grid,
     .          lo_sub(nferdims), hi_sub(nferdims), axis
	REAL	bad_com, bad_res, lo_com, hi_com, unit, cos_factor,
     .          ybxlo, ybxhi, bx
	REAL*8	TM_WORLD, lo_ww, hi_ww

* internal (convenience) equivalences
	INTEGER	lo_s1, lo_s2, lo_s3, lo_s4, lo_s5, lo_s6,
     .          hi_s1, hi_s2, hi_s3, hi_s4, hi_s5, hi_s6
	EQUIVALENCE (lo_s1, lo_sub(1)) , (lo_s2, lo_sub(2)),
     .		    (lo_s3, lo_sub(3)) , (lo_s4, lo_sub(4)),
     .		    (lo_s5, lo_sub(5)) , (lo_s6, lo_sub(6)),
     .		    (hi_s1, hi_sub(1)) , (hi_s2, hi_sub(2)),
     .		    (hi_s3, hi_sub(3)) , (hi_s4, hi_sub(4)),
     .		    (hi_s5, hi_sub(5)) , (hi_s6, hi_sub(6))

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing', isact_class_trans, res_cx, idim )

* initialize
	grid    = mr_grid( res_mr )
	axis    = grid_line( idim, grid )
	bad_com = mr_bad_data( res_mr )
	bad_res = mr_bad_data( com_mr )

* compute limits within which calculation is possible
	DO ii = 1, nferdims
	   lo_sub(ii) = cx_lo_ss(res_cx,ii)
	   hi_sub(ii) = cx_hi_ss(res_cx,ii)
	ENDDO
	lo_sub(idim) = com_lo + 1 
	hi_sub(idim) = com_hi - 1

* create an array of delta H increments
	DO ii = lo_sub(idim), hi_sub(idim)
	   lo_ww = TM_WORLD( ii-1, grid, idim, box_middle )
	   hi_ww = TM_WORLD( ii+1, grid, idim, box_middle )
	   box(ii) = hi_ww - lo_ww
	ENDDO	   

* perform differentiation in standard units if possible
	unit = un_convert( line_unit_code(axis) )
	IF (ITSA_TRUEMONTH_AXIS(axis)) unit = un_convert(pun_day)
	DO ii = lo_sub(idim), hi_sub(idim)
	   box(ii) = box(ii) * unit
 	ENDDO

* DERIVATIVE ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   need_cos = GEOG_COS_FACTOR( idim, grid )
     .                .AND. lo_s2.NE.unspecified_int4
	   DO j = lo_s2, hi_s2
	   IF ( need_cos ) THEN
	      cos_factor = COS( deg2rad *       
     .            TM_WORLD(j, grid, y_dim, box_middle) ) 
              ybxlo = deg2rad* 
     .            TM_WORLD(j,grid,y_dim,box_lo_lim)
              ybxhi = deg2rad* 
     .            TM_WORLD(j,grid,y_dim,box_hi_lim)
              bx = ABS(ybxhi - ybxlo)
	      cos_factor = ABS( SIN(ybxhi) - SIN(ybxlo) )/ bx
	   ENDIF
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	      DO i = lo_s1, hi_s1
	         lo_com = com(i-1,j,k,l,m,n)
	         hi_com = com(i+1,j,k,l,m,n)
	         IF ( lo_com.EQ.bad_res .OR. hi_com.EQ.bad_res ) THEN
	            res(i,j,k,l,m,n) = bad_res
	         ELSE
	            res(i,j,k,l,m,n) = (hi_com-lo_com) / box(i)
	            IF ( need_cos )
     .                 res(i,j,k,l,m,n) = res(i,j,k,l,m,n)/cos_factor
	         ENDIF
	      ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* DERIVATIVE ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO i = lo_s1, hi_s1
	      DO j = lo_s2, hi_s2
	         lo_com = com(i,j-1,k,l,m,n)
	         hi_com = com(i,j+1,k,l,m,n)
	         IF ( lo_com.EQ.bad_res .OR. hi_com.EQ.bad_res ) THEN
	            res(i,j,k,l,m,n) = bad_res
	         ELSE
	            res(i,j,k,l,m,n) = (hi_com-lo_com) / box(j)
	         ENDIF
	      ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* DERIVATIVE ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      DO k = lo_s3, hi_s3
	         lo_com = com(i,j,k-1,l,m,n)
	         hi_com = com(i,j,k+1,l,m,n)
	         IF ( lo_com.EQ.bad_res .OR. hi_com.EQ.bad_res ) THEN
	            res(i,j,k,l,m,n) = bad_res
	         ELSE
	            res(i,j,k,l,m,n) = (hi_com-lo_com) / box(k)
	         ENDIF
	      ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* DERIVATIVE ALONG T AXIS
	ELSEIF ( idim .EQ. t_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      DO l = lo_s4, hi_s4
	         lo_com = com(i,j,k,l-1,m,n)
	         hi_com = com(i,j,k,l+1,m,n)
	         IF ( lo_com.EQ.bad_res .OR. hi_com.EQ.bad_res ) THEN
	            res(i,j,k,l,m,n) = bad_res
	         ELSE
	            res(i,j,k,l,m,n) = (hi_com-lo_com) / box(l)
	         ENDIF
	      ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* DERIVATIVE ALONG E AXIS
	ELSEIF ( idim .EQ. e_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      DO m = lo_s5, hi_s5
	         lo_com = com(i,j,k,l,m-1,n)
	         hi_com = com(i,j,k,l,m+1,n)
	         IF ( lo_com.EQ.bad_res .OR. hi_com.EQ.bad_res ) THEN
	            res(i,j,k,l,m,n) = bad_res
	         ELSE
	            res(i,j,k,l,m,n) = (hi_com-lo_com) / box(m)
	         ENDIF
	      ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* DERIVATIVE ALONG F AXIS
	ELSEIF ( idim .EQ. f_dim ) THEN
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      DO n = lo_s6, hi_s6
	         lo_com = com(i,j,k,l,m,n-1)
	         hi_com = com(i,j,k,l,m,n+1)
	         IF ( lo_com.EQ.bad_res .OR. hi_com.EQ.bad_res ) THEN
	            res(i,j,k,l,m,n) = bad_res
	         ELSE
	            res(i,j,k,l,m,n) = (hi_com-lo_com) / box(n)
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
