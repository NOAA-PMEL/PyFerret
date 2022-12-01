	INTEGER FUNCTION DO_FILL_NEAR(  idim, arg,
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
* fill along axis idim by duplicating the closest available valid pt

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V???:  3/23/99 - based on DO_FILL_AVE
*         7/2/99 - changed logic so if 2 equi-distant points use the mean
*                - corrected subscriptong bug in L axis search
* V550 *sh* 8/02 - support for modulo lengths. ensure that filling 
*		   does not operate acros the subspan modulo void point
*	   11/02 - change no_point to a very negative nalue (was -1)
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)

* Note:  this routine could be made simpler and faster by arranging single
*        inner loops that looped "loop=1,edge" looking at index-loop and
*        index+loop to find the first valid point(s)
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
	LOGICAL	TM_ITS_SUBSPAN_MODULO, has_mod_void, TM_ITSA_DSG
	INTEGER TM_MODULO_LINE_DIM, CGRID_AXIS,
     .          TM_DSG_DSET_FROM_GRID, TM_DSG_NFEATURES, 
     .          DO_FILL_NEAR_DSG,
     .		i, j, k, l, m, n, ii, jj, kk, ll, mm, nn, 
     .          lo_sub(nferdims), hi_sub(nferdims),
     .		lo_edge, hi_edge, lo_pos, hi_pos, edge,
     .		npts, line, 
     .		dset, grid, nfeatures, maxobs, mr_list(2)
	REAL	bad_res, lo_val, hi_val, little_val, big_val

* internal (convenience) equivalences
	INTEGER	lo_s1, lo_s2, lo_s3, lo_s4, lo_s5, lo_s6,
     .          hi_s1, hi_s2, hi_s3, hi_s4, hi_s5, hi_s6
	EQUIVALENCE (lo_s1, lo_sub(1)) , (lo_s2, lo_sub(2)),
     .		    (lo_s3, lo_sub(3)) , (lo_s4, lo_sub(4)),
     .		    (lo_s5, lo_sub(5)) , (lo_s6, lo_sub(6)),
     .		    (hi_s1, hi_sub(1)) , (hi_s2, hi_sub(2)),
     .		    (hi_s3, hi_sub(3)) , (hi_s4, hi_sub(4)),
     .		    (hi_s5, hi_sub(5)) , (hi_s6, hi_sub(6))

* local parameter declarations
	INTEGER     no_point
	PARAMETER ( no_point = -99999 )

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing', isact_class_trans, res_cx, idim )

* initialize
	bad_res = mr_bad_data( com_mr )
	edge = arg
	line = CGRID_AXIS(idim, com_cx)
	has_mod_void = TM_ITS_SUBSPAN_MODULO(line)
	npts = TM_MODULO_LINE_DIM(line)


* DSG-specific initialize

	grid = mr_grid( com_mr )
	IF (TM_ITSA_DSG(grid)) THEN 

* DSG ragged array
	   dset = TM_DSG_DSET_FROM_GRID(grid)
	   nfeatures = TM_DSG_NFEATURES(grid)
* ... get the length of the longest feature
	   CALL DSG_COORD_LIMS(dset, pdsg_row_size, little_val, big_val)
	   maxobs = big_val

	   mr_list(1) = com_mr
	   mr_list(2) = res_mr
	   CALL NON_ARRAY_SUBSC( mr_list, 2 )
	   CALL DSG_ARRAY_SUBSC( mr_list, 2, dsg_orientation(dset) )

	   DO_FILL_NEAR = DO_FILL_NEAR_DSG ( 
     .				idim, dset, nfeatures, maxobs,
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx)

	   GOTO 5000
	ENDIF

* compute limits within which calculation is straightforward
	DO ii = 1, nferdims
	   lo_sub(ii) = cx_lo_ss(res_cx,ii)
	   hi_sub(ii) = cx_hi_ss(res_cx,ii)
	ENDDO
	lo_edge = MAX( lo_sub(idim)-edge, cx_lo_ss(com_cx, idim) )
	hi_edge = MIN( hi_sub(idim)+edge, cx_hi_ss(com_cx, idim) )

* FILL ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2

           lo_pos = no_point
           hi_pos = no_point
           DO i = lo_edge, lo_s1
              IF ( com(i,j,k,l,m,n) .NE. bad_res ) THEN
                 lo_val = com(i,j,k,l,m,n)
                 lo_pos = i
              ENDIF
	   ENDDO

	   DO i = lo_s1, hi_s1
	      IF ( com(i,j,k,l,m,n) .NE. bad_res ) THEN
	         res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
                 lo_val = com(i,j,k,l,m,n)
                 lo_pos = i
	         CYCLE
	      ENDIF
* ... look forward for next valid point
              IF ( hi_pos .LT. i ) THEN
                 DO ii = i+1, hi_edge
                    IF ( com(ii,j,k,l,m,n) .NE. bad_res ) THEN
                       hi_pos = ii
                       hi_val = com(ii,j,k,l,m,n)
                       GOTO 108
                    ENDIF
                    hi_pos = no_point
		 ENDDO
              ENDIF
* ... choose closest valid point
 108          CONTINUE
              IF ( has_mod_void ) THEN
*    ... forbid crossing a subspan modulo void point
	         IF (lo_pos.NE.no_point) THEN
	           IF (i/npts .NE. lo_pos/npts) lo_pos=no_point
	         ENDIF
	         IF (hi_pos.NE.no_point) THEN
	           IF (i/npts .NE. hi_pos/npts) hi_pos=no_point
	         ENDIF
	      ENDIF

	      IF ( lo_pos.EQ.no_point ) THEN
                 IF ( hi_pos.EQ.no_point ) THEN
                    res(i,j,k,l,m,n) = bad_res
                 ELSE
                    res(i,j,k,l,m,n) = hi_val
                 ENDIF
              ELSE
                 IF ( hi_pos.EQ.no_point ) THEN
                    res(i,j,k,l,m,n) = lo_val
                 ELSEIF ( i-lo_pos .EQ. hi_pos-i ) THEN
                    res(i,j,k,l,m,n) = ( lo_val + hi_val ) /2.
                 ELSEIF ( i-lo_pos .LT. hi_pos-i ) THEN
                    res(i,j,k,l,m,n) = lo_val
                 ELSE
                    res(i,j,k,l,m,n) = hi_val
                 ENDIF
              ENDIF
	      IF (has_mod_void) THEN
	         IF (MOD(i,npts) .EQ. 0)
     .              res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
	      ENDIF

 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

* FILL ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO i = lo_s1, hi_s1
           lo_pos = no_point
           hi_pos = no_point

           DO j = lo_edge, lo_s2
              IF ( com(i,j,k,l,m,n) .NE. bad_res ) THEN
                 lo_val = com(i,j,k,l,m,n)
                 lo_pos = j
              ENDIF
           ENDDO

	   DO j = lo_s2, hi_s2
	      IF ( com(i,j,k,l,m,n) .NE. bad_res ) THEN
	         res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
                 lo_val = com(i,j,k,l,m,n)
                 lo_pos = j
	         CYCLE
	      ENDIF
* ... look forward for next valid point
              IF ( hi_pos .LT. j ) THEN
                 DO jj = j+1, hi_edge
                    IF ( com(i,jj,k,l,m,n) .NE. bad_res ) THEN
                       hi_pos = jj
                       hi_val = com(i,jj,k,l,m,n)
                       GOTO 208
                    ENDIF
                 ENDDO 
                 hi_pos = no_point
              ENDIF
* ... choose closest valid point
 208          IF ( has_mod_void ) THEN
*    ... forbid crossing a subspan modulo void point
	         IF (lo_pos.NE.no_point) THEN
	           IF (j/npts .NE. lo_pos/npts) lo_pos=no_point
	         ENDIF
	         IF (hi_pos.NE.no_point) THEN
	           IF (j/npts .NE. hi_pos/npts) hi_pos=no_point
	         ENDIF
	      ENDIF

             IF ( lo_pos.EQ.no_point ) THEN
                 IF ( hi_pos.EQ.no_point ) THEN
                    res(i,j,k,l,m,n) = bad_res
                 ELSE
                    res(i,j,k,l,m,n) = hi_val
                 ENDIF
              ELSE
                 IF ( hi_pos.EQ.no_point ) THEN
                    res(i,j,k,l,m,n) = lo_val
                 ELSEIF ( j-lo_pos .EQ. hi_pos-j ) THEN
                    res(i,j,k,l,m,n) = ( lo_val + hi_val ) /2.
                 ELSEIF ( j-lo_pos .LT. hi_pos-j ) THEN
                    res(i,j,k,l,m,n) = lo_val
                 ELSE
                    res(i,j,k,l,m,n) = hi_val
                 ENDIF
              ENDIF
	      IF (has_mod_void) THEN
	         IF (MOD(j,npts) .EQ. 0)
     .              res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
	      ENDIF

 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

* FILL ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
           lo_pos = no_point
           hi_pos = no_point
           DO k = lo_edge, lo_s3
              IF ( com(i,j,k,l,m,n) .NE. bad_res ) THEN
                 lo_val = com(i,j,k,l,m,n)
                 lo_pos = k
              ENDIF
           ENDDO
	   DO k = lo_s3, hi_s3
	      IF ( com(i,j,k,l,m,n) .NE. bad_res ) THEN
	         res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
                 lo_val = com(i,j,k,l,m,n)
                 lo_pos = k
	         CYCLE
              ENDIF
* ... look forward for next valid point
              IF ( hi_pos .LT. k ) THEN
                 DO kk = k+1, hi_edge
                    IF ( com(i,j,kk,l,m,n) .NE. bad_res ) THEN
                       hi_pos = kk
                       hi_val = com(i,j,kk,l,m,n)
                       GOTO 308
                    ENDIF
                 ENDDO
                 hi_pos = no_point
              ENDIF
* ... choose closest valid point
 308          IF ( has_mod_void ) THEN
*    ... forbid crossing a subspan modulo void point
	         IF (lo_pos.NE.no_point) THEN
	           IF (k/npts .NE. lo_pos/npts) lo_pos=no_point
	         ENDIF
	         IF (hi_pos.NE.no_point) THEN
	           IF (k/npts .NE. hi_pos/npts) hi_pos=no_point
	         ENDIF
	      ENDIF

              IF ( lo_pos.EQ.no_point ) THEN
                 IF ( hi_pos.EQ.no_point ) THEN
                    res(i,j,k,l,m,n) = bad_res
                 ELSE
                    res(i,j,k,l,m,n) = hi_val
                 ENDIF
              ELSE
                 IF ( hi_pos.EQ.no_point ) THEN
                    res(i,j,k,l,m,n) = lo_val
                 ELSEIF ( k-lo_pos .EQ. hi_pos-k ) THEN
                    res(i,j,k,l,m,n) = ( lo_val + hi_val ) /2.
                 ELSEIF ( k-lo_pos .LT. hi_pos-k ) THEN
                    res(i,j,k,l,m,n) = lo_val
                 ELSE
                    res(i,j,k,l,m,n) = hi_val
                 ENDIF
              ENDIF
	      IF (has_mod_void) THEN
	         IF (MOD(k,npts) .EQ. 0)
     .              res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
	      ENDIF

 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

* FILL ALONG T AXIS
	ELSEIF ( idim .EQ. t_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
           lo_pos = no_point
           hi_pos = no_point
           DO l = lo_edge, lo_s4
              IF ( com(i,j,k,l,m,n) .NE. bad_res ) THEN
                 lo_val = com(i,j,k,l,m,n)
                 lo_pos = l
              ENDIF
           ENDDO
	   DO l = lo_s4, hi_s4
	      IF ( com(i,j,k,l,m,n) .NE. bad_res ) THEN
	         res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
                 lo_val = com(i,j,k,l,m,n)
                 lo_pos = l
	         CYCLE
	      ENDIF
              IF ( hi_pos .LT. l ) THEN
* ... look forward for next valid point
                 DO ll = l+1, hi_edge
                    IF ( com(i,j,k,ll,m,n) .NE. bad_res ) THEN
                       hi_pos = ll
                       hi_val = com(i,j,k,ll,m,n)
                       GOTO 408
                    ENDIF
                 hi_pos = no_point
                 ENDDO
              ENDIF
* ... choose closest valid point
 408          IF ( has_mod_void ) THEN
*    ... forbid crossing a subspan modulo void point
	         IF (lo_pos.NE.no_point) THEN
	           IF (l/npts .NE. lo_pos/npts) lo_pos=no_point
	         ENDIF
	         IF (hi_pos.NE.no_point) THEN
	           IF (l/npts .NE. hi_pos/npts) hi_pos=no_point
	         ENDIF
	      ENDIF

              IF ( lo_pos.EQ.no_point ) THEN
                 IF ( hi_pos.EQ.no_point ) THEN
                    res(i,j,k,l,m,n) = bad_res
                 ELSE
                    res(i,j,k,l,m,n) = hi_val
                 ENDIF
              ELSE
                 IF ( hi_pos.EQ.no_point ) THEN
                    res(i,j,k,l,m,n) = lo_val
                 ELSEIF ( l-lo_pos .EQ. hi_pos-l ) THEN
                    res(i,j,k,l,m,n) = ( lo_val + hi_val ) /2.
                 ELSEIF ( l-lo_pos .LT. hi_pos-l ) THEN
                    res(i,j,k,l,m,n) = lo_val
                 ELSE
                    res(i,j,k,l,m,n) = hi_val
                 ENDIF
              ENDIF
	      IF (has_mod_void) THEN
	         IF (MOD(l,npts) .EQ. 0)
     .              res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
	      ENDIF

 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

* FILL ALONG E AXIS
	ELSEIF ( idim .EQ. e_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
           lo_pos = no_point
           hi_pos = no_point
           DO m = lo_edge, lo_s5
              IF ( com(i,j,k,l,m,n) .NE. bad_res ) THEN
                 lo_val = com(i,j,k,l,m,n)
                 lo_pos = m
              ENDIF
           ENDDO
	   DO m = lo_s5, hi_s5
	      IF ( com(i,j,k,l,m,n) .NE. bad_res ) THEN
	         res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
                 lo_val = com(i,j,k,l,m,n)
                 lo_pos = m
	         CYCLE
	      ENDIF
              IF ( hi_pos .LT. m ) THEN
* ... Look forward for next valid point
                 DO mm = m+1, hi_edge
                    IF ( com(i,j,k,l,mm,n) .NE. bad_res ) THEN
                       hi_pos = mm
                       hi_val = com(i,j,k,l,mm,n)
                       GOTO 508
                    ENDIF
                 ENDDO
                 hi_pos = no_point
              ENDIF
* ... choose closest valid point
 508          IF ( has_mod_void ) THEN
*    ... forbid crossing a subspan modulo void point
	         IF (lo_pos.NE.no_point) THEN
	           IF (m/npts .NE. lo_pos/npts) lo_pos=no_point
	         ENDIF
	         IF (hi_pos.NE.no_point) THEN
	           IF (m/npts .NE. hi_pos/npts) hi_pos=no_point
	         ENDIF
	      ENDIF

              IF ( lo_pos.EQ.no_point ) THEN
                 IF ( hi_pos.EQ.no_point ) THEN
                    res(i,j,k,l,m,n) = bad_res
                 ELSE
                    res(i,j,k,l,m,n) = hi_val
                 ENDIF
              ELSE
                 IF ( hi_pos.EQ.no_point ) THEN
                    res(i,j,k,l,m,n) = lo_val
                 ELSEIF ( m-lo_pos .EQ. hi_pos-m ) THEN
                    res(i,j,k,l,m,n) = ( lo_val + hi_val ) /2.
                 ELSEIF ( m-lo_pos .LT. hi_pos-m ) THEN
                    res(i,j,k,l,m,n) = lo_val
                 ELSE
                    res(i,j,k,l,m,n) = hi_val
                 ENDIF
              ENDIF
	      IF (has_mod_void) THEN
	         IF (MOD(m,npts) .EQ. 0)
     .              res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
	      ENDIF

 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

* FILL ALONG F AXIS
	ELSEIF ( idim .EQ. f_dim ) THEN
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
           lo_pos = no_point
           hi_pos = no_point
           DO n = lo_edge, lo_s6
              IF ( com(i,j,k,l,m,n) .NE. bad_res ) THEN
                 lo_val = com(i,j,k,l,m,n)
                 lo_pos =n
              ENDIF
           ENDDO
	   DO n = lo_s6, hi_s6
	      IF ( com(i,j,k,l,m,n) .NE. bad_res ) THEN
	         res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
                 lo_val = com(i,j,k,l,m,n)
                 lo_pos = n
	         CYCLE
	      ENDIF
              IF ( hi_pos .LT. n ) THEN
* ... Look forward for next valid point
                 DO nn = n+1, hi_edge
                    IF ( com(i,j,k,l,m,nn) .NE. bad_res ) THEN
                       hi_pos = nn
                       hi_val = com(i,j,k,l,m,nn)
                       GOTO 608
                    ENDIF
                 ENDDO
                 hi_pos = no_point
              ENDIF
* ... choose closest valid point
 608          IF ( has_mod_void ) THEN
*    ... forbid crossing a subspan modulo void point
	         IF (lo_pos.NE.no_point) THEN
	           IF (n/npts .NE. lo_pos/npts) lo_pos=no_point
	         ENDIF
	         IF (hi_pos.NE.no_point) THEN
	           IF (n/npts .NE. hi_pos/npts) hi_pos=no_point
	         ENDIF
	      ENDIF

              IF ( lo_pos.EQ.no_point ) THEN
                 IF ( hi_pos.EQ.no_point ) THEN
                    res(i,j,k,l,m,n) = bad_res
                 ELSE
                    res(i,j,k,l,m,n) = hi_val
                 ENDIF
              ELSE
                 IF ( hi_pos.EQ.no_point ) THEN
                    res(i,j,k,l,m,n) = lo_val
                 ELSEIF ( n-lo_pos .EQ. hi_pos-n ) THEN
                    res(i,j,k,l,m,n) = ( lo_val + hi_val ) /2.
                 ELSEIF ( n-lo_pos .LT. hi_pos-n ) THEN
                    res(i,j,k,l,m,n) = lo_val
                 ELSE
                    res(i,j,k,l,m,n) = hi_val
                 ENDIF
              ENDIF
	      IF (has_mod_void) THEN
	         IF (MOD(n,npts) .EQ. 0)
     .              res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
	      ENDIF

 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

	ENDIF

* always success
 5000	DO_FILL_NEAR = ferr_ok
	
	RETURN

	END
