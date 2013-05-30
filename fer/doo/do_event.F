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
        LOGICAL first_pt, first_lt
* *kob* 6.01 - step_val needs to be integer
	INTEGER step_val, i, j, k, l, m, n,
     .          lo_sub(nferdims), hi_sub(nferdims)
	REAL	bad_com

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
	bad_com = mr_bad_data( com_mr )
        
* calculation limits
	DO 10 i = 1, nferdims
	   lo_sub(i) = cx_lo_ss(res_cx,i)
	   hi_sub(i) = cx_hi_ss(res_cx,i)
 10	CONTINUE

* SEARCH ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO 110 n = lo_s6, hi_s6
	   DO 110 m = lo_s5, hi_s5
	   DO 110 l = lo_s4, hi_s4
	   DO 110 k = lo_s3, hi_s3
	   DO 110 j = lo_s2, hi_s2
	      step_val = 0
              first_pt = .TRUE.
	      DO 100 i = lo_s1, hi_s1
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
 100          CONTINUE
 110       CONTINUE

* SEARCH ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO 210 n = lo_s6, hi_s6
	   DO 210 m = lo_s5, hi_s5
	   DO 210 l = lo_s4, hi_s4
	   DO 210 k = lo_s3, hi_s3
	   DO 210 i = lo_s1, hi_s1
	      step_val = 0
              first_pt = .TRUE.
	      DO 200 j = lo_s2, hi_s2
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
 200         CONTINUE
 210      CONTINUE

* SEARCH ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 310 n = lo_s6, hi_s6
	   DO 310 m = lo_s5, hi_s5
	   DO 310 l = lo_s4, hi_s4
	   DO 310 j = lo_s2, hi_s2
	   DO 310 i = lo_s1, hi_s1
	      step_val = 0
              first_pt = .TRUE.
	      DO 300 k = lo_s3, hi_s3
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
 300         CONTINUE
 310      CONTINUE

* SEARCH ALONG T AXIS
	ELSEIF ( idim .EQ. t_dim ) THEN
	   DO 410 n = lo_s6, hi_s6
	   DO 410 m = lo_s5, hi_s5
	   DO 410 k = lo_s3, hi_s3
	   DO 410 j = lo_s2, hi_s2
	   DO 410 i = lo_s1, hi_s1
	      step_val = 0
              first_pt = .TRUE.
	      DO 400 l = lo_s4, hi_s4
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
 400         CONTINUE
 410      CONTINUE

* SEARCH ALONG E AXIS
	ELSEIF ( idim .EQ. e_dim ) THEN
	   DO 510 n = lo_s6, hi_s6
	   DO 510 l = lo_s4, hi_s4
	   DO 510 k = lo_s3, hi_s3
	   DO 510 j = lo_s2, hi_s2
	   DO 510 i = lo_s1, hi_s1
	      step_val = 0
              first_pt = .TRUE.
	      DO 500 m = lo_s5, hi_s5
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
 500         CONTINUE
 510      CONTINUE

* SEARCH ALONG F AXIS
	ELSEIF ( idim .EQ. f_dim ) THEN
	   DO 610 m = lo_s5, hi_s5
	   DO 610 l = lo_s4, hi_s4
	   DO 610 k = lo_s3, hi_s3
	   DO 610 j = lo_s2, hi_s2
	   DO 610 i = lo_s1, hi_s1
	      step_val = 0
              first_pt = .TRUE.
	      DO 600 n = lo_s6, hi_s6
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
 600         CONTINUE
 610      CONTINUE

	ENDIF

* always success
	DO_EVENT = ferr_ok
	RETURN

	END
