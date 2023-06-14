	INTEGER FUNCTION DO_FILL_AVE(	idim, arg,
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
* fill along axis idim by a running mean window

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/2/89
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN
* V550 *sh* 8/2/02 - support for modulo lengths. ensure that filling 
*		    does not operate acros the subspan modulo void point
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
	INTEGER	TM_MODULO_LINE_DIM, CGRID_AXIS, 
     .          TM_DSG_DSET_FROM_GRID, TM_DSG_NFEATURES, 
     .          DO_FILL_AVE_DSG,
     .		i, j, k, l, m, n, ii, cnt, haf,
     .          lo_sub(nferdims), hi_sub(nferdims),
     .		ii_com, lo_com, hi_com, lo_com0, hi_com0,
     .		lo_modlim, hi_modlim, nmod, nmod_prev, modlen, 
     .		dset, grid, nfeatures, maxobs, mr_list(2)

	REAL	bad_res, sum, comp, little_val, big_val

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
	haf = arg * .5
	bad_res = mr_bad_data( com_mr )
	nmod_prev = -99999

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

	   DO_FILL_AVE = DO_FILL_AVE_DSG ( 
     .				idim, haf, dset, nfeatures, maxobs,
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx)
	   GOTO 5000
	ENDIF


* compute limits within which calculation is straightforward
	DO ii = 1, nferdims
	   lo_sub(ii) = cx_lo_ss(res_cx,ii)
	   hi_sub(ii) = cx_hi_ss(res_cx,ii)
	ENDDO
	lo_com0 = cx_lo_ss(com_cx,idim)
	hi_com0 = cx_hi_ss(com_cx,idim)
	lo_com = lo_com0
	hi_com = hi_com0

* it his a subspan modulo axis?
	has_mod_void = TM_ITS_SUBSPAN_MODULO( CGRID_AXIS(idim,com_cx) )
	IF (has_mod_void) THEN
	   modlen = TM_MODULO_LINE_DIM( CGRID_AXIS(idim,com_cx) )
	   has_mod_void = lo_com.LE.0 .OR. hi_com.GE.modlen
	ENDIF

* FILL ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
*  ... do not allow filling based upon values across a subspan modulo void
	      IF (has_mod_void) THEN
	         IF (MOD(i,modlen) .EQ. 0) THEN
	            res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
	            CYCLE
	         ENDIF
	         nmod = (i-1)/modlen 
	         IF (i.LE.0) nmod = nmod - 1
	         IF ( nmod .NE. nmod_prev ) THEN
	            nmod_prev = nmod
	            lo_modlim = nmod*modlen + 1
	            hi_modlim = (nmod+1) * modlen
	            lo_com = MAX( lo_modlim, lo_com0 )
	            hi_com = MIN( hi_modlim, hi_com0 )
	         ENDIF
	      ENDIF
* ... end of subspan modulo adjustment
	      IF ( com(i,j,k,l,m,n) .NE. bad_res ) THEN
	         res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
	         CYCLE
	      ENDIF
	      cnt = 0
	      sum = 0.0
	      DO ii = -haf, haf
	         ii_com = i + ii
	         IF ( ii_com.LT.lo_com .OR. ii_com.GT.hi_com ) CYCLE
	         comp = com(ii_com,j,k,l,m,n)
	         IF ( comp .NE. bad_res ) THEN
	            cnt = cnt + 1
	            sum = sum + comp
	         ENDIF
	      ENDDO
	      
	      IF ( cnt .GT. 0 ) THEN
	         res(i,j,k,l,m,n) = sum / cnt
	      ELSE
	         res(i,j,k,l,m,n) = bad_res
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
	   DO j = lo_s2, hi_s2
*  ... do not allow filling based upon values across a subspan modulo void
	      IF (has_mod_void) THEN
	         IF (MOD(j,modlen) .EQ. 0) THEN
	            res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
	            CYCLE
	         ENDIF
	         nmod = (j-1)/modlen 
	         IF (j.LE.0) nmod = nmod - 1
	         IF ( nmod .NE. nmod_prev ) THEN
	            nmod_prev = nmod
	            lo_modlim = nmod*modlen + 1
	            hi_modlim = (nmod+1) * modlen
	            lo_com = MAX( lo_modlim, lo_com0 )
	            hi_com = MIN( hi_modlim, hi_com0 )
	         ENDIF
	      ENDIF
* ... end of subspan modulo adjustment
	   DO i = lo_s1, hi_s1
	      IF ( com(i,j,k,l,m,n) .NE. bad_res ) THEN
	         res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
	         CYCLE
	      ENDIF
	      cnt = 0
	      sum = 0.0
	      DO ii = -haf, haf
	         ii_com = j + ii
	         IF ( ii_com.LT.lo_com .OR. ii_com.GT.hi_com ) CYCLE
	         comp = com(i,ii_com,k,l,m,n)
	         IF ( comp .NE. bad_res ) THEN
	            cnt = cnt + 1
	            sum = sum + comp
	         ENDIF
	      ENDDO
	      IF ( cnt .GT. 0 ) THEN
	         res(i,j,k,l,m,n) = sum / cnt
	      ELSE
	         res(i,j,k,l,m,n) = bad_res
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
	   DO k = lo_s3, hi_s3
*  ... do not allow filling based upon values across a subspan modulo void
	      IF (has_mod_void) THEN
	         IF (MOD(k,modlen) .EQ. 0) THEN
	            res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
	            CYCLE
	         ENDIF
	         nmod = (k-1)/modlen 
	         IF (k.LE.0) nmod = nmod - 1
	         IF ( nmod .NE. nmod_prev ) THEN
	            nmod_prev = nmod
	            lo_modlim = nmod*modlen + 1
	            hi_modlim = (nmod+1) * modlen
	            lo_com = MAX( lo_modlim, lo_com0 )
	            hi_com = MIN( hi_modlim, hi_com0 )
	         ENDIF
	      ENDIF
* ... end of subspan modulo adjustment
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      IF ( com(i,j,k,l,m,n) .NE. bad_res ) THEN
	         res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
	         CYCLE
	      ENDIF
	      cnt = 0
	      sum = 0.0
	      DO ii = -haf, haf
	         ii_com = k + ii
	         IF ( ii_com.LT.lo_com .OR. ii_com.GT.hi_com ) CYCLE
	         comp = com(i,j,ii_com,l,m,n)
	         IF ( comp .NE. bad_res ) THEN
	            cnt = cnt + 1
	            sum = sum + comp
	         ENDIF
	      ENDDO
	      IF ( cnt .GT. 0 ) THEN
	         res(i,j,k,l,m,n) = sum / cnt
	      ELSE
	         res(i,j,k,l,m,n) = bad_res
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
	   DO l = lo_s4, hi_s4
*  ... do not allow filling based upon values across a subspan modulo void
	      IF (has_mod_void) THEN
	         IF (MOD(l,modlen) .EQ. 0) THEN
	            res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
	            CYCLE
	         ENDIF
	         nmod = (l-1)/modlen 
	         IF (l.LE.0) nmod = nmod - 1
	         IF ( nmod .NE. nmod_prev ) THEN
	            nmod_prev = nmod
	            lo_modlim = nmod*modlen + 1
	            hi_modlim = (nmod+1) * modlen
	            lo_com = MAX( lo_modlim, lo_com0 )
	            hi_com = MIN( hi_modlim, hi_com0 )
	         ENDIF
	      ENDIF
* ... end of subspan modulo adjustment
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      IF ( com(i,j,k,l,m,n) .NE. bad_res ) THEN
	         res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
	         CYCLE
	      ENDIF
	      cnt = 0
	      sum = 0.0
	      DO ii = -haf, haf
	         ii_com = l + ii
	         IF ( ii_com.LT.lo_com .OR. ii_com.GT.hi_com ) CYCLE
	         comp = com(i,j,k,ii_com,m,n)
	         IF ( comp .NE. bad_res ) THEN
	            cnt = cnt + 1
	            sum = sum + comp
	         ENDIF
	      ENDDO
	      IF ( cnt .GT. 0 ) THEN
	         res(i,j,k,l,m,n) = sum / cnt
	      ELSE
	         res(i,j,k,l,m,n) = bad_res
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
	   DO m = lo_s5, hi_s5
*  ... do not allow filling based upon values across a subspan modulo void
	      IF (has_mod_void) THEN
	         IF (MOD(m,modlen) .EQ. 0) THEN
	            res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
	            CYCLE
	         ENDIF
	         nmod = (m-1)/modlen 
	         IF (m.LE.0) nmod = nmod - 1
	         IF ( nmod .NE. nmod_prev ) THEN
	            nmod_prev = nmod
	            lo_modlim = nmod*modlen + 1
	            hi_modlim = (nmod+1) * modlen
	            lo_com = MAX( lo_modlim, lo_com0 )
	            hi_com = MIN( hi_modlim, hi_com0 )
	         ENDIF
	      ENDIF
* ... end of subspan modulo adjustment
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      IF ( com(i,j,k,l,m,n) .NE. bad_res ) THEN
	         res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
	         CYCLE
	      ENDIF
	      cnt = 0
	      sum = 0.0
	      DO ii = -haf, haf
	         ii_com = m + ii
	         IF ( ii_com.LT.lo_com .OR. ii_com.GT.hi_com ) CYCLE
	         comp = com(i,j,k,l,ii_com,n)
	         IF ( comp .NE. bad_res ) THEN
	            cnt = cnt + 1
	            sum = sum + comp
	         ENDIF
	      ENDDO
	      IF ( cnt .GT. 0 ) THEN
	         res(i,j,k,l,m,n) = sum / cnt
	      ELSE
	         res(i,j,k,l,m,n) = bad_res
	      ENDIF
	   ENDDO
	   ENDDO

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* FILL ALONG F AXIS
	ELSEIF ( idim .EQ. f_dim ) THEN
	   DO n = lo_s6, hi_s6
*  ... do not allow filling based upon values across a subspan modulo void
	      IF (has_mod_void) THEN
	         IF (MOD(n,modlen) .EQ. 0) THEN
	            res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
	            CYCLE
	         ENDIF
	         nmod = (n-1)/modlen 
	         IF (n.LE.0) nmod = nmod - 1
	         IF ( nmod .NE. nmod_prev ) THEN
	            nmod_prev = nmod
	            lo_modlim = nmod*modlen + 1
	            hi_modlim = (nmod+1) * modlen
	            lo_com = MAX( lo_modlim, lo_com0 )
	            hi_com = MIN( hi_modlim, hi_com0 )
	         ENDIF
	      ENDIF
* ... end of subspan modulo adjustment
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      IF ( com(i,j,k,l,m,n) .NE. bad_res ) THEN
	         res(i,j,k,l,m,n) = com(i,j,k,l,m,n)
	         CYCLE
	      ENDIF
	      cnt = 0
	      sum = 0.0
	      DO ii = -haf, haf
	         ii_com = n + ii
	         IF ( ii_com.LT.lo_com .OR. ii_com.GT.hi_com ) CYCLE
	         comp = com(i,j,k,l,m,ii_com)
	         IF ( comp .NE. bad_res ) THEN
	            cnt = cnt + 1
	            sum = sum + comp
	         ENDIF
	      ENDDO
	      IF ( cnt .GT. 0 ) THEN
	         res(i,j,k,l,m,n) = sum / cnt
	      ELSE
	         res(i,j,k,l,m,n) = bad_res
	      ENDIF
	   ENDDO

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	ENDIF

* always success
 5000	DO_FILL_AVE = ferr_ok
	RETURN

	END

