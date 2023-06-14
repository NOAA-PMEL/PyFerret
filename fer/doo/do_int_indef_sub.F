	SUBROUTINE DO_INT_INDEF_SUB (idim, arg,
     .					   com_lo, com_hi,
     .					   com, com_mr, com_cx,
     .					   res, res_mr, res_cx,
     .					   box	)

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
* integrate the component along axis idim producing an "indefinite" integral

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:	10/11/89 - from old DO_INT_INDEF
*	  2/7/90 - eliminated diagnostic output - duplicated in calling routine
* V230: 5/6/92  - COS correction checks that J location is defined
*       8/10/92 - results were wrong for integration within a single box
* 9/09 - Apply fix to the cos factor (previous versions of the fix had not
*        been made), bug1348
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.
* v741 *acm*  5/18 remove old unused ifdef
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xprog_state.cmn'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xunits.cmn_text'
	external xunits_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data

* calling argument declarations:
	INTEGER	idim, com_lo, com_hi, com_mr, com_cx, res_mr, res_cx
	REAL	arg,
     .          com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
     .		box( com_lo:com_hi )

* internal variable declarations:
	LOGICAL	GEOG_COS_FACTOR, ITSA_TRUEMONTH_AXIS, 
     .          need_cos
	INTEGER i, j, k, l, m, n, grid, lo, hi, axis,
     .          ii, jj, kk, ll, mm, nn
	REAL	BOX_SIZE, unit, bad_com, bad_res, cos_factor,
     .          ybxlo, ybxhi, bx
	REAL*8	TM_WORLD, dsum

* initialize
	grid = mr_grid( com_mr )
	axis = grid_line( idim, grid )
	lo   = cx_lo_ss( com_cx, idim )
	hi   = cx_hi_ss( com_cx, idim )
	bad_com = mr_bad_data( com_mr )
	bad_res = mr_bad_data( res_mr )

* get and save grid box sizes (weights) just once for efficiency
* (end boxes may be only partially weighted)
        IF ( lo .EQ. hi ) THEN
           box(lo) = cx_hi_ww(idim,res_cx) - cx_lo_ww(idim,res_cx)
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
	DO i = lo, hi
	   box(i) = box(i) * unit
	ENDDO

* INTEGRATE
* (is double precision summing necessary ? *sh* 6/89)
* ALONG X AXIS
* (note: X integration is a special case because it requires COSINE(lat) )
	IF ( idim .EQ. x_dim ) THEN
	   need_cos = GEOG_COS_FACTOR( idim, grid )
     .                .AND. cx_lo_s2(com_cx).NE.unspecified_int4
	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
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
	   DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	      dsum = 0.0D0
	      DO i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
* ... reject all data after a bad point
	            DO ii = i, hi
		       res(ii,j,k,l,m,n) = bad_res
	            ENDDO
		    GOTO 210
	         ENDIF
	         dsum = dsum + box(i)*com(i,j,k,l,m,n)
	         IF ( need_cos ) THEN
	            res(i,j,k,l,m,n) = dsum * cos_factor
	         ELSE
	            res(i,j,k,l,m,n) = dsum
	         ENDIF
	      ENDDO
 210	      CONTINUE
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO  n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO  m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO  l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO  k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO  i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	      dsum = 0.0D0
	      DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
* ... reject all data after a bad point
	            DO jj = j, hi
		       res(i,jj,k,l,m,n) = bad_res
	            ENDDO
		    GOTO 310
	         ENDIF
	         dsum = dsum + box(j)*com(i,j,k,l,m,n)
	         res(i,j,k,l,m,n) = dsum
	      ENDDO
 310	   CONTINUE
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
	      dsum = 0.0D0
	      DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
* ... reject all data after a bad point
	            DO kk = k, hi
		       res(i,j,kk,l,m,n) = bad_res
	            ENDDO
		    GOTO 410
	         ENDIF
	         dsum = dsum + box(k)*com(i,j,k,l,m,n)
	         res(i,j,k,l,m,n) = dsum
	      ENDDO
 410	   CONTINUE
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
	      dsum = 0.0D0
	      DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
* ... reject all data after a bad point
	            DO ll = l, hi
		       res(i,j,k,ll,m,n) = bad_res
	            ENDDO
		    GOTO 510
	         ENDIF
	         dsum = dsum + box(l)*com(i,j,k,l,m,n)
	         res(i,j,k,l,m,n) = dsum
	      ENDDO
 510	   CONTINUE
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
	      dsum = 0.0D0
	      DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
* ... reject all data after a bad point
	            DO mm = m, hi
		       res(i,j,k,l,mm,n) = bad_res
	            ENDDO
		    GOTO 610
	         ENDIF
	         dsum = dsum + box(m)*com(i,j,k,l,m,n)
	         res(i,j,k,l,m,n) = dsum
	      ENDDO
 610	   CONTINUE
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
	      dsum = 0.0D0
	      DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
* ... reject all data after a bad point
	            DO nn = n, hi
		       res(i,j,k,l,m,nn) = bad_res
	            ENDDO
		    GOTO 710
	         ENDIF
	         dsum = dsum + box(n)*com(i,j,k,l,m,n)
	         res(i,j,k,l,m,n) = dsum
	      ENDDO
 710	   CONTINUE
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	ENDIF

	RETURN
	END
