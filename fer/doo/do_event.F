	INTEGER FUNCTION DO_EVENT(	idim, arg,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx,
     .					buff	)

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
* the indicated argument value is regarded as an "event threshhold".
* Any time that the component variable crosses or lands on this value
* an "event" occurs and the output should step up by "1" to indicate it.
* If an exact match occurs, step up ON that value.
* If a crossing occurs, step up on the first value AFTER the crossing.  

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V533:  6/01 - new routine based on DO_DIST_LOC
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* v685   5/13 *acm* Fix ticket 2054, EVNT in z direction - typo in equiv.
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
	include	'xdset_info.cmn_text'

* calling argument declarations:
	INTEGER	idim, com_mr, com_cx, res_mr, res_cx
	REAL	arg, buff(*),
     .          com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif )

* internal variable declarations:
        LOGICAL TM_ITSA_DSG, first_pt, first_lt
* *kob* 6.01 - step_val needs to be integer
	INTEGER TM_DSG_DSET_FROM_GRID, TM_DSG_NFEATURES,
     .		step_val, i, j, k, l, m, n, grid, dset, 
     .		nfeatures, orientation, maxobs, mr_list(2),
     .          lo_sub(nferdims), hi_sub(nferdims)
	REAL	bad_com
        REAL    little_val, big_val


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


* DSGs are handled as a separate case
	grid = cx_grid(com_cx)
	IF ( TM_ITSA_DSG(grid) ) THEN
	   dset = TM_DSG_DSET_FROM_GRID(grid)
	   nfeatures = TM_DSG_NFEATURES(grid)
	   orientation = dsg_orientation(dset)
	   mr_list(1) = com_mr
	   mr_list(2) = res_mr
	   CALL NON_ARRAY_SUBSC( mr_list, 2 )
	   CALL DSG_ARRAY_SUBSC( mr_list, 2, orientation )
	   
* ... get the length of the longest feature
           CALL DSG_COORD_LIMS(dset, pdsg_row_size, little_val, big_val)
           maxobs = big_val

	   CALL DO_EVENT_DSG( idim, arg, dset, nfeatures, maxobs,
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx )

	   GOTO 5000
	ENDIF

* initialize
	bad_com = mr_bad_data( com_mr )
        
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
	      step_val = 0
              first_pt = .TRUE.
	      DO i = lo_s1, hi_s1
                 IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
	            first_pt = .TRUE.
                 ELSEIF ( com(i,j,k,l,m,n) .EQ. arg ) THEN
* ... exact equality
	            step_val = step_val + 1
	            first_pt = .TRUE.
                 ELSEIF ( first_pt ) THEN
* ... first non-bad value
                    first_pt = .FALSE.
                    first_lt = com(i,j,k,l,m,n) .LT. arg
                 ELSEIF ( first_lt .NEQV.
     .                    (com(i,j,k,l,m,n) .LT. arg) ) THEN
* ... found a "crossing"
	            step_val = step_val + 1
	            first_lt = .NOT. first_lt
                 ENDIF
		 res(i,j,k,l,m,n) = step_val
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
	      step_val = 0
              first_pt = .TRUE.
	      DO j = lo_s2, hi_s2
                 IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
	            first_pt = .TRUE.
                 ELSEIF ( com(i,j,k,l,m,n) .EQ. arg ) THEN
* ... exact equality
	            step_val = step_val + 1
	            first_pt = .TRUE.
                 ELSEIF ( first_pt ) THEN
* ... first non-bad value
                    first_pt = .FALSE.
                    first_lt = com(i,j,k,l,m,n) .LT. arg
                 ELSEIF ( first_lt .NEQV.
     .                    (com(i,j,k,l,m,n) .LT. arg) ) THEN
* ... found a "crossing"
	            step_val = step_val + 1
	            first_lt = .NOT. first_lt
                 ENDIF
		 res(i,j,k,l,m,n) = step_val
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
	      step_val = 0
              first_pt = .TRUE.
	      DO k = lo_s3, hi_s3
                 IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
	            first_pt = .TRUE.
                 ELSEIF ( com(i,j,k,l,m,n) .EQ. arg ) THEN
* ... exact equality
	            step_val = step_val + 1
	            first_pt = .TRUE.
                 ELSEIF ( first_pt ) THEN
* ... first non-bad value
                    first_pt = .FALSE.
                    first_lt = com(i,j,k,l,m,n) .LT. arg
                 ELSEIF ( first_lt .NEQV.
     .                    (com(i,j,k,l,m,n) .LT. arg) ) THEN
* ... found a "crossing"
	            step_val = step_val + 1
	            first_lt = .NOT. first_lt
                 ENDIF
		 res(i,j,k,l,m,n) = step_val
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
	      step_val = 0
              first_pt = .TRUE.
	      DO l = lo_s4, hi_s4
                 IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
	            first_pt = .TRUE.
                 ELSEIF ( com(i,j,k,l,m,n) .EQ. arg ) THEN
* ... exact equality
	            step_val = step_val + 1
	            first_pt = .TRUE.
                 ELSEIF ( first_pt ) THEN
* ... first non-bad value
                    first_pt = .FALSE.
                    first_lt = com(i,j,k,l,m,n) .LT. arg
                 ELSEIF ( first_lt .NEQV.
     .                    (com(i,j,k,l,m,n) .LT. arg) ) THEN
* ... found a "crossing"
	            step_val = step_val + 1
	            first_lt = .NOT. first_lt
                 ENDIF
		 res(i,j,k,l,m,n) = step_val
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
	      step_val = 0
              first_pt = .TRUE.
	      DO m = lo_s5, hi_s5
                 IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
	            first_pt = .TRUE.
                 ELSEIF ( com(i,j,k,l,m,n) .EQ. arg ) THEN
* ... exact equality
	            step_val = step_val + 1
	            first_pt = .TRUE.
                 ELSEIF ( first_pt ) THEN
* ... first non-bad value
                    first_pt = .FALSE.
                    first_lt = com(i,j,k,l,m,n) .LT. arg
                 ELSEIF ( first_lt .NEQV.
     .                    (com(i,j,k,l,m,n) .LT. arg) ) THEN
* ... found a "crossing"
	            step_val = step_val + 1
	            first_lt = .NOT. first_lt
                 ENDIF
		 res(i,j,k,l,m,n) = step_val
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
	      step_val = 0
              first_pt = .TRUE.
	      DO n = lo_s6, hi_s6
                 IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
	            first_pt = .TRUE.
                 ELSEIF ( com(i,j,k,l,m,n) .EQ. arg ) THEN
* ... exact equality
	            step_val = step_val + 1
	            first_pt = .TRUE.
                 ELSEIF ( first_pt ) THEN
* ... first non-bad value
                    first_pt = .FALSE.
                    first_lt = com(i,j,k,l,m,n) .LT. arg
                 ELSEIF ( first_lt .NEQV.
     .                    (com(i,j,k,l,m,n) .LT. arg) ) THEN
* ... found a "crossing"
	            step_val = step_val + 1
	            first_lt = .NOT. first_lt
                 ENDIF
		 res(i,j,k,l,m,n) = step_val
	      ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	ENDIF

* always success
 5000	DO_EVENT = ferr_ok
	RETURN

	END
