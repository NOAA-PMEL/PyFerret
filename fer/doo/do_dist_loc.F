	INTEGER FUNCTION DO_DIST_LOC(	idim, arg,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx,
     .					buff	)

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
* locate the indicated argument value along the indicated axis and leave
* behind an "interpolating kernel": linear interpolating weights on the
* points immediately before and after the located value

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V230:  1/22/92 - new routine based on DO_FILL_AVE and DO_LOCATE
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
	INTEGER	idim, com_mr, com_cx, res_mr, res_cx
	REAL	arg, buff(*),
     .          com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif )

* internal variable declarations:
        LOGICAL first_pt, first_lt, crossing_found
	INTEGER i, j, k, l, m, n, ss1,
     .          lo_sub(nferdims), hi_sub(nferdims)
	REAL	bad_res, com1, com2, del

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
	bad_res = mr_bad_data( com_mr )
        
* calculation limits
	DO i = 1, nferdims
	   lo_sub(i) = cx_lo_ss(res_cx,i)
	   hi_sub(i) = cx_hi_ss(res_cx,i)
	ENDDO

* SEARCH ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
              first_pt = .TRUE.
              crossing_found = .FALSE.
	      DO i = lo_s1, hi_s1
                 res(i,j,k,l,m,n) = bad_res
                 IF ( crossing_found ) CYCLE
                 IF ( com(i,j,k,l,m,n) .EQ. bad_res ) CYCLE
                 IF ( com(i,j,k,l,m,n) .EQ. arg ) THEN
* ... exact equality
                    res(i,j,k,l,m,n) = 1.0
                    crossing_found = .TRUE.
                 ELSEIF ( first_pt ) THEN
* ... first non-bad value
                    first_pt = .FALSE.
                    first_lt = com(i,j,k,l,m,n) .LT. arg
                    ss1 = i
                 ELSEIF ( first_lt .EQV.
     .                    (com(i,j,k,l,m,n) .LT. arg) ) THEN
* ... still have not crossed from GT to LT or visa versa
                    ss1 = i
                 ELSE
* ... found a "crossing"
                    com1 = com(ss1,j,k,l,m,n)
                    com2 = com(i  ,j,k,l,m,n)
                    del = com2 - com1
                    res(ss1,j,k,l,m,n) = (com2-arg) / del
                    res(i  ,j,k,l,m,n) = (arg-com1) / del
                    crossing_found = .TRUE.
                 ENDIF
	      ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* SEARCH ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO i = lo_s1, hi_s1
              first_pt = .TRUE.
              crossing_found = .FALSE.
	      DO j = lo_s2, hi_s2
                 res(i,j,k,l,m,n) = bad_res
                 IF ( crossing_found ) CYCLE
                 IF ( com(i,j,k,l,m,n) .EQ. bad_res ) CYCLE
                 IF ( com(i,j,k,l,m,n) .EQ. arg ) THEN
* ... exact equality
                    res(i,j,k,l,m,n) = 1.0
                    crossing_found = .TRUE.
                 ELSEIF ( first_pt ) THEN
* ... first non-bad value
                    first_pt = .FALSE.
                    first_lt = com(i,j,k,l,m,n) .LT. arg
                    ss1 = j
                 ELSEIF ( first_lt .EQV.
     .                    (com(i,j,k,l,m,n) .LT. arg) ) THEN
* ... still have not crossed from GT to LT or visa versa
                    ss1 = j
                 ELSE
* ... found a "crossing"
                    com1 = com(i,ss1,k,l,m,n)
                    com2 = com(i,j  ,k,l,m,n)
                    del = com2 - com1
                    res(i,ss1,k,l,m,n) = (com2-arg) / del
                    res(i,j  ,k,l,m,n) = (arg-com1) / del
                    crossing_found = .TRUE.
                ENDIF
	      ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* SEARCH ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
              first_pt = .TRUE.
              crossing_found = .FALSE.
	      DO k = lo_s3, hi_s3
                 res(i,j,k,l,m,n) = bad_res
                 IF ( crossing_found ) CYCLE
                 IF ( com(i,j,k,l,m,n) .EQ. bad_res ) CYCLE
                 IF ( com(i,j,k,l,m,n) .EQ. arg ) THEN
* ... exact equality
                    res(i,j,k,l,m,n) = 1.0
                    crossing_found = .TRUE.
                 ELSEIF ( first_pt ) THEN
* ... first non-bad value
                    first_pt = .FALSE.
                    first_lt = com(i,j,k,l,m,n) .LT. arg
                    ss1 = k
                 ELSEIF ( first_lt .EQV.
     .                    (com(i,j,k,l,m,n) .LT. arg) ) THEN
* ... still have not crossed from GT to LT or visa versa
                    ss1 = k
                 ELSE
* ... found a "crossing"
                    com1 = com(i,j,ss1,l,m,n)
                    com2 = com(i,j,k  ,l,m,n)
                    del = com2 - com1
                    res(i,j,ss1,l,m,n) = (com2-arg) / del
                    res(i,j,k,  l,m,n) = (arg-com1) / del
                    crossing_found = .TRUE.
                ENDIF
	      ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* SEARCH ALONG T AXIS
	ELSEIF ( idim .EQ. t_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
              first_pt = .TRUE.
              crossing_found = .FALSE.
	      DO l = lo_s4, hi_s4
                 res(i,j,k,l,m,n) = bad_res
                 IF ( crossing_found ) CYCLE
                 IF ( com(i,j,k,l,m,n) .EQ. bad_res ) CYCLE
                 IF ( com(i,j,k,l,m,n) .EQ. arg ) THEN
* ... exact equality
                    res(i,j,k,l,m,n) = 1.0
                    crossing_found = .TRUE.
                 ELSEIF ( first_pt ) THEN
* ... first non-bad value
                    first_pt = .FALSE.
                    first_lt = com(i,j,k,l,m,n) .LT. arg
                    ss1 = l
                 ELSEIF ( first_lt .EQV.
     .                    (com(i,j,k,l,m,n) .LT. arg) ) THEN
* ... still have not crossed from GT to LT or visa versa
                    ss1 = l
                 ELSE
* ... found a "crossing"
                    com1 = com(i,j,k,ss1,m,n)
                    com2 = com(i,j,k,l,  m,n)
                    del = com2 - com1
                    res(i,j,k,ss1,m,n) = (com2-arg) / del
                    res(i,j,k,l,  m,n) = (arg-com1) / del
                    crossing_found = .TRUE.
                ENDIF
	      ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* SEARCH ALONG E AXIS
	ELSEIF ( idim .EQ. e_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
              first_pt = .TRUE.
              crossing_found = .FALSE.
	      DO m = lo_s5, hi_s5
                 res(i,j,k,l,m,n) = bad_res
                 IF ( crossing_found ) CYCLE
                 IF ( com(i,j,k,l,m,n) .EQ. bad_res ) CYCLE
                 IF ( com(i,j,k,l,m,n) .EQ. arg ) THEN
* ... exact equality
                    res(i,j,k,l,m,n) = 1.0
                    crossing_found = .TRUE.
                 ELSEIF ( first_pt ) THEN
* ... first non-bad value
                    first_pt = .FALSE.
                    first_lt = com(i,j,k,l,m,n) .LT. arg
                    ss1 = m
                 ELSEIF ( first_lt .EQV.
     .                    (com(i,j,k,l,m,n) .LT. arg) ) THEN
* ... still have not crossed from GT to LT or visa versa
                    ss1 = m
                 ELSE
* ... found a "crossing"
                    com1 = com(i,j,k,l,ss1,n)
                    com2 = com(i,j,k,l,  m,n)
                    del = com2 - com1
                    res(i,j,k,l,ss1,n) = (com2-arg) / del
                    res(i,j,k,l,  m,n) = (arg-com1) / del
                    crossing_found = .TRUE.
                ENDIF
	      ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* SEARCH ALONG F AXIS
	ELSEIF ( idim .EQ. f_dim ) THEN
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
              first_pt = .TRUE.
              crossing_found = .FALSE.
	      DO n = lo_s6, hi_s6
                 res(i,j,k,l,m,n) = bad_res
                 IF ( crossing_found ) CYCLE
                 IF ( com(i,j,k,l,m,n) .EQ. bad_res ) CYCLE
                 IF ( com(i,j,k,l,m,n) .EQ. arg ) THEN
* ... exact equality
                    res(i,j,k,l,m,n) = 1.0
                    crossing_found = .TRUE.
                 ELSEIF ( first_pt ) THEN
* ... first non-bad value
                    first_pt = .FALSE.
                    first_lt = com(i,j,k,l,m,n) .LT. arg
                    ss1 = n
                 ELSEIF ( first_lt .EQV.
     .                    (com(i,j,k,l,m,n) .LT. arg) ) THEN
* ... still have not crossed from GT to LT or visa versa
                    ss1 = n
                 ELSE
* ... found a "crossing"
                    com1 = com(i,j,k,l,m,ss1)
                    com2 = com(i,j,k,l,m,  n)
                    del = com2 - com1
                    res(i,j,k,l,m,ss1) = (com2-arg) / del
                    res(i,j,k,l,m,  n) = (arg-com1) / del
                    crossing_found = .TRUE.
                ENDIF
	      ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	ENDIF

* always success
	DO_DIST_LOC = ferr_ok
	RETURN

	END
