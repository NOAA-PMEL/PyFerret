	INTEGER FUNCTION DO_INTERPOLATE( axis, arg,
     .					 com, com_mr, com_cx,
     .					 res, res_mr, res_cx,
     .					 work	)

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
* interpolate the component grid along the given axis
* note: if requested interpolation lies beyond the first or last grid point
*	(but within the edge grid box) then the component context will be
*	only that single edge point (ss_lo=ss_hi).  The code below will
*	extrapolate the end value to the box edge.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/1/89
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
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

* calling argument declarations:
	INTEGER	axis, com_mr, com_cx, res_mr, res_cx
	REAL	arg, work(*),
     .          com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif )

* internal variable declarations:
	INTEGER i, j, k, l, m, n, ss, ss_lo, ss_hi
	REAL	bad, delta, c_lo, c_hi, z_lo, z_hi
	REAL*8	TM_WORLD, ww, ww_lo, ww_hi

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing', isact_class_trans, res_cx, axis )

* flags for bad/missing data
        bad = mr_bad_data(res_mr)

* bits and pieces for interpolation
	ss    = cx_lo_ss(res_cx,axis)
	ww    = cx_lo_ww(axis,res_cx)
	ss_lo = cx_lo_ss(com_cx,axis)
	ss_hi = cx_hi_ss(com_cx,axis)
	ww_lo = TM_WORLD( ss_lo, cx_grid(com_cx), axis, box_middle )
	ww_hi = TM_WORLD( ss_hi, cx_grid(com_cx), axis, box_middle )
	delta = ww_hi - ww_lo
	IF ( delta .EQ. 0.0 ) THEN
* ... extrapolate last grid point towards end box edge
	   c_lo = 1.0
	   c_hi = 0.0
	ELSE
* ... normal interpolation
	   c_lo    = REAL(ww_hi - ww) / delta
	   c_hi    = 1.0 - c_lo
	ENDIF

*       I N T E R P O L A T E
* ( separate loops for each axis )
	IF ( axis .EQ. x_dim ) THEN

	   DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)

* source data points
	      z_lo = com(ss_lo,j,k,l,m,n)
	      z_hi = com(ss_hi,j,k,l,m,n)

* interpolate
	      IF (z_lo .EQ. bad .OR. z_hi .EQ. bad ) THEN
	         res(ss,j,k,l,m,n) = bad
	      ELSE
	         res(ss,j,k,l,m,n) = c_lo*z_lo + c_hi*z_hi
	      ENDIF

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	ELSEIF ( axis .EQ. y_dim ) THEN

	   DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)

* source data points
	      z_lo = com(i,ss_lo,k,l,m,n)
	      z_hi = com(i,ss_hi,k,l,m,n)

* interpolate
	      IF (z_lo .EQ. bad .OR. z_hi .EQ. bad ) THEN
	         res(i,ss,k,l,m,n) = bad
	      ELSE
	         res(i,ss,k,l,m,n) = c_lo*z_lo + c_hi*z_hi
	      ENDIF

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	ELSEIF ( axis .EQ. z_dim ) THEN

	   DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)

* source data points
	      z_lo = com(i,j,ss_lo,l,m,n)
	      z_hi = com(i,j,ss_hi,l,m,n)

* interpolate
	      IF (z_lo .EQ. bad .OR. z_hi .EQ. bad ) THEN
	         res(i,j,ss,l,m,n) = bad
	      ELSE
	         res(i,j,ss,l,m,n) = c_lo*z_lo + c_hi*z_hi
	      ENDIF

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	ELSEIF ( axis .EQ. t_dim ) THEN

	   DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)

* source data points
	      z_lo = com(i,j,k,ss_lo,m,n)
	      z_hi = com(i,j,k,ss_hi,m,n)

* interpolate
	      IF (z_lo .EQ. bad .OR. z_hi .EQ. bad ) THEN
	         res(i,j,k,ss,m,n) = bad
	      ELSE
	         res(i,j,k,ss,m,n) = c_lo*z_lo + c_hi*z_hi
	      ENDIF

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	ELSEIF ( axis .EQ. e_dim ) THEN

	   DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)

* source data points
	      z_lo = com(i,j,k,l,ss_lo,n)
	      z_hi = com(i,j,k,l,ss_hi,n)

* interpolate
	      IF (z_lo .EQ. bad .OR. z_hi .EQ. bad ) THEN
	         res(i,j,k,l,ss,n) = bad
	      ELSE
	         res(i,j,k,l,ss,n) = c_lo*z_lo + c_hi*z_hi
	      ENDIF

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	ELSEIF ( axis .EQ. f_dim ) THEN

	   DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)

* source data points
	      z_lo = com(i,j,k,l,m,ss_lo)
	      z_hi = com(i,j,k,l,m,ss_hi)

* interpolate
	      IF (z_lo .EQ. bad .OR. z_hi .EQ. bad ) THEN
	         res(i,j,k,l,m,ss) = bad
	      ELSE
	         res(i,j,k,l,m,ss) = c_lo*z_lo + c_hi*z_hi
	      ENDIF

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	ENDIF

	DO_INTERPOLATE = ferr_ok

	RETURN
	END
