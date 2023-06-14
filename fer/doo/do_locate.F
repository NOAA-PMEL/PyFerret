	INTEGER FUNCTION DO_LOCATE( idim, arg,
     .				 com, com_mr, com_cx,
     .				 res, res_mr, res_cx,
     .				 buff	)

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
* find the world coordinate at which the component grid equals the argument

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  1/11/90
* V230:  1/22/92 - fixed bug of missing exact equality
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
	LOGICAL	first_pt, first_lt
	INTEGER i, j, k, l, m, n, pt, ss1, grid
	REAL	bad_res, ww1, ww2, del, com1, com2
	REAL*8	TM_WORLD

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing', isact_class_trans, res_cx, idim )

* initialize
	grid = cx_grid( com_cx )
	pt   = cx_lo_ss( res_cx, idim )
	bad_res = mr_bad_data( res_mr )

* ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	      first_pt = .TRUE.
	      DO i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_res ) CYCLE
                 IF ( com(i,j,k,l,m,n) .EQ. arg ) THEN
                    res(pt,j,k,l,m,n) = TM_WORLD(i, grid, idim, box_middle)
	            GOTO 110
	         ELSEIF ( first_pt ) THEN
	           first_pt = .FALSE.
		   first_lt = com(i,j,k,l,m,n) .LT. arg
	           ss1 = i
		 ELSEIF ( first_lt .EQV. (com(i,j,k,l,m,n).LT.arg) ) THEN
	           ss1 = i
		 ELSE
* ... found a "zero-crossing"
	            ww1 = TM_WORLD ( ss1, grid, idim, box_middle )
	            ww2 = TM_WORLD ( i,   grid, idim, box_middle )
	            com1 = com(ss1,j,k,l,m,n)
	            com2 = com(i  ,j,k,l,m,n)
	            del = com2 - com1
	            res(pt,j,k,l,m,n) =
     .                 ( ww2*(arg-com1) + ww1*(com2-arg) ) / del
	            GOTO 110
	         ENDIF
	      ENDDO
* ... no zero-crossing found
	      res(pt,j,k,l,m,n) = bad_res
 110	   CONTINUE
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
	      first_pt = .TRUE.
	      DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_res ) CYCLE
                 IF ( com(i,j,k,l,m,n) .EQ. arg) THEN
	            res(i,pt,k,l,m,n) = TM_WORLD( j, grid, idim, box_middle )
	            GOTO 210
	         ELSEIF ( first_pt ) THEN
	           first_pt = .FALSE.
		   first_lt = com(i,j,k,l,m,n) .LT. arg
	           ss1 = j
		 ELSEIF ( first_lt .EQV. (com(i,j,k,l,m,n).LT.arg) ) THEN
	           ss1 = j
		 ELSE
* ... found a "zero-crossing"
	            ww1 = TM_WORLD ( ss1, grid, idim, box_middle )
	            ww2 = TM_WORLD ( j,   grid, idim, box_middle )
	            com1 = com(i,ss1,k,l,m,n)
	            com2 = com(i,j  ,k,l,m,n)
	            del = com2 - com1
	            res(i,pt,k,l,m,n) =
     .                 ( ww2*(arg-com1) + ww1*(com2-arg) ) / del
	            GOTO 210
	         ENDIF
	      ENDDO
* ... no zero-crossing found
	      res(i,pt,k,l,m,n) = bad_res
 210	   CONTINUE
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
	      first_pt = .TRUE.
	      DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_res ) CYCLE
                 IF ( com(i,j,k,l,m,n) .EQ. arg ) THEN
	            res(i,j,pt,l,m,n) = TM_WORLD ( k, grid, idim, box_middle )
	            GOTO 310
	         ELSEIF ( first_pt ) THEN
	           first_pt = .FALSE.
		   first_lt = com(i,j,k,l,m,n) .LT. arg
	           ss1 = k
		 ELSEIF ( first_lt .EQV. (com(i,j,k,l,m,n).LT.arg) ) THEN
	           ss1 = k
		 ELSE
* ... found a "zero-crossing"
	            ww1 = TM_WORLD ( ss1, grid, idim, box_middle )
	            ww2 = TM_WORLD ( k,   grid, idim, box_middle )
	            com1 = com(i,j,ss1,l,m,n)
	            com2 = com(i,j,k  ,l,m,n)
	            del = com2 - com1
	            res(i,j,pt,l,m,n) =
     .                 ( ww2*(arg-com1) + ww1*(com2-arg) ) / del
	            GOTO 310
	         ENDIF
	      ENDDO
* ... no zero-crossing found
	      res(i,j,pt,l,m,n) = bad_res
 310	   CONTINUE
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
	      first_pt = .TRUE.
	      DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_res ) CYCLE
                 IF ( com(i,j,k,l,m,n) .EQ. arg ) THEN
	            res(i,j,k,pt,m,n) = TM_WORLD( l, grid, idim, box_middle )
	            GOTO 410
	         ELSEIF ( first_pt ) THEN
	           first_pt = .FALSE.
		   first_lt = com(i,j,k,l,m,n) .LT. arg
	           ss1 = l
		 ELSEIF ( first_lt .EQV. (com(i,j,k,l,m,n).LT.arg) ) THEN
	           ss1 = l
		 ELSE
* ... found a "zero-crossing"
	            ww1 = TM_WORLD ( ss1, grid, idim, box_middle )
	            ww2 = TM_WORLD ( l,   grid, idim, box_middle )
	            com1 = com(i,j,k,ss1,m,n)
	            com2 = com(i,j,k,l,  m,n)
	            del = com2 - com1
	            res(i,j,k,pt,m,n) =
     .                 ( ww2*(arg-com1) + ww1*(com2-arg) ) / del
	            GOTO 410
	         ENDIF
	      ENDDO
* ... no zero-crossing found
	      res(i,j,k,pt,m,n) = bad_res
 410	   CONTINUE
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
	      first_pt = .TRUE.
	      DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_res ) CYCLE
                 IF ( com(i,j,k,l,m,n) .EQ. arg ) THEN
	            res(i,j,k,l,pt,n) = TM_WORLD( m, grid, idim, box_middle )
	            GOTO 510
	         ELSEIF ( first_pt ) THEN
	           first_pt = .FALSE.
		   first_lt = com(i,j,k,l,m,n) .LT. arg
	           ss1 = m
		 ELSEIF ( first_lt .EQV. (com(i,j,k,l,m,n).LT.arg) ) THEN
	           ss1 = m
		 ELSE
* ... found a "zero-crossing"
	            ww1 = TM_WORLD ( ss1, grid, idim, box_middle )
	            ww2 = TM_WORLD ( m,   grid, idim, box_middle )
	            com1 = com(i,j,k,l,ss1,n)
	            com2 = com(i,j,k,l,  m,n)
	            del = com2 - com1
	            res(i,j,k,l,pt,n) =
     .                 ( ww2*(arg-com1) + ww1*(com2-arg) ) / del
	            GOTO 510
	         ENDIF
	      ENDDO
* ... no zero-crossing found
	      res(i,j,k,l,pt,n) = bad_res
 510	   CONTINUE
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
	      first_pt = .TRUE.
	      DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_res ) CYCLE
                 IF ( com(i,j,k,l,m,n) .EQ. arg ) THEN
	            res(i,j,k,l,m,pt) = TM_WORLD( n, grid, idim, box_middle )
	            GOTO 610
	         ELSEIF ( first_pt ) THEN
	           first_pt = .FALSE.
		   first_lt = com(i,j,k,l,m,n) .LT. arg
	           ss1 = n
		 ELSEIF ( first_lt .EQV. (com(i,j,k,l,m,n).LT.arg) ) THEN
	           ss1 = n
		 ELSE
* ... found a "zero-crossing"
	            ww1 = TM_WORLD ( ss1, grid, idim, box_middle )
	            ww2 = TM_WORLD ( n,   grid, idim, box_middle )
	            com1 = com(i,j,k,l,m,ss1)
	            com2 = com(i,j,k,l,m,n)
	            del = com2 - com1
	            res(i,j,k,l,m,pt) =
     .                 ( ww2*(arg-com1) + ww1*(com2-arg) ) / del
	            GOTO 610
	         ENDIF
	      ENDDO
* ... no zero-crossing found
	      res(i,j,k,l,m,pt) = bad_res
 610	   CONTINUE
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

	ENDIF

	DO_LOCATE = ferr_ok

	RETURN
	END
