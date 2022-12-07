	SUBROUTINE MEDIAN(	idim, wlen, 
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx, 
     .				list )

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
* Compute the median of the requested length, along axis idim
* note: the component context may not be of adequate size for the full
*	calculation.  Missing data flags will be inserted where computation is
*	impossible. If there is any data present in the window, the median
*       is computed based on that data. Medians are computed with the data
*       that is available in partial windows at the edges of the domain. 

* programmer - ansley manke based on convolve.
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'

* calling argument declarations:
	INTEGER	idim, wlen, com_mr, com_cx, res_mr, res_cx
	REAL	com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
     .          list(*)

* internal variable declarations:
	LOGICAL TM_ITS_SUBSPAN_MODULO, has_mod_void
	INTEGER	TM_MODULO_LINE_DIM, CGRID_AXIS, 
     .		i, j, k, l, m, n, ii, ndx, lo_lim, hi_lim,
     .          lo_sub(nferdims), hi_sub(nferdims), hlen,
     .		lo_modlim, hi_modlim, nmod, nmod_prev, 
     .		modlen, median_loc
	REAL    bad_com, bad_res, comp

* internal (convenience) equivalences
	INTEGER	lo_s1, lo_s2, lo_s3, lo_s4, lo_s5, lo_s6,
     .          hi_s1, hi_s2, hi_s3, hi_s4, hi_s5, hi_s6
	EQUIVALENCE (lo_s1, lo_sub(1)) , (lo_s2, lo_sub(2)),
     .		    (lo_s3, lo_sub(3)) , (lo_s4, lo_sub(4)),
     .		    (lo_s5, lo_sub(5)) , (lo_s6, lo_sub(6)),
     .		    (hi_s1, hi_sub(1)) , (hi_s2, hi_sub(2)),
     .		    (hi_s3, hi_sub(3)) , (hi_s4, hi_sub(4)),
     .		    (hi_s5, hi_sub(5)) , (hi_s6, hi_sub(6))

* initialize
	bad_com = mr_bad_data( res_mr )
	bad_res = mr_bad_data( com_mr )
	lo_lim = cx_lo_ss(com_cx,idim)
	hi_lim = cx_hi_ss(com_cx,idim) 
	nmod_prev = -99999

	hlen = wlen/2
	median_loc = hlen

* compute limits within which calculation is possible
	DO ii = 1, nferdims
	   lo_sub(ii) = cx_lo_ss(res_cx,ii)
	   hi_sub(ii) = cx_hi_ss(res_cx,ii)
	ENDDO

* it this a subspan modulo axis?
	has_mod_void = TM_ITS_SUBSPAN_MODULO( CGRID_AXIS(idim,com_cx) )
	IF (has_mod_void) THEN
	   modlen = TM_MODULO_LINE_DIM( CGRID_AXIS(idim,com_cx) )
	   has_mod_void = lo_lim.LE.0 .OR. hi_lim.GE.modlen
	ENDIF

* ... though treating each axis separately is to avoid testing inside the loop
* MEDIAN ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      IF (has_mod_void) THEN
*  ... do not allow computation across the modulo void
	         IF (i.GE.1) THEN  ! get # of modulos to curr index
	            nmod = (i-1)/modlen 
	         ELSE
	            nmod = i/modlen - 1
	         ENDIF
	         IF ( nmod .NE. nmod_prev ) THEN
	            nmod_prev = nmod
	            lo_modlim = nmod*modlen + 1
	            hi_modlim = (nmod+1) * modlen
	            lo_lim = MAX( lo_modlim, cx_lo_ss(com_cx,idim) )
	            hi_lim = MIN( hi_modlim, cx_hi_ss(com_cx,idim)  )
	         ENDIF
	      ENDIF
	      ndx = 1
	      DO ii = -hlen, hlen
* test limits of component data before using
                 IF ( i+ii .GE. lo_lim
     .          .AND. i+ii .LE. hi_lim ) THEN
                    comp = com(i+ii,j,k,l,m,n)
                 ELSE
                    comp = bad_com
                 ENDIF

	         IF ( comp .NE. bad_com ) THEN
	            list(ndx) = comp
		    ndx = ndx + 1
	         ENDIF
	      ENDDO

              ndx = ndx - 1
	      IF (ndx .EQ. 0) THEN 
                res(i,j,k,l,m,n) = bad_res
                CYCLE
	      ENDIF

	      IF (ndx .EQ. 1) THEN 
                res(i,j,k,l,m,n) = list(1)
                CYCLE
	      ENDIF
              CALL HEAP2_V(list, bad_com, ndx)

              median_loc = ndx/2 + 1
	      res(i,j,k,l,m,n) = list(median_loc)

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* MEDIAN ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	      IF (has_mod_void) THEN
*  ... do not allow computation across the modulo void
	         IF (j.GE.1) THEN  ! get # of modulos to curr index
	            nmod = (j-1)/modlen 
	         ELSE
	            nmod = j/modlen - 1
	         ENDIF
	         IF ( nmod .NE. nmod_prev ) THEN
	            nmod_prev = nmod
	            lo_modlim = nmod*modlen + 1
	            hi_modlim = (nmod+1) * modlen
	            lo_lim = MAX( lo_modlim, cx_lo_ss(com_cx,idim) )
	            hi_lim = MIN( hi_modlim, cx_hi_ss(com_cx,idim)  )
	         ENDIF
	      ENDIF
	   DO i = lo_s1, hi_s1
	      ndx = 1
	      DO ii = -hlen, hlen
* test limits of component data before using
                 IF ( j+ii .GE. lo_lim
     .          .AND. j+ii .LE. hi_lim ) THEN
                    comp = com(i,j+ii,k,l,m,n)
                 ELSE
                    comp = bad_com
                 ENDIF

	         IF ( comp .NE. bad_com ) THEN
	            list(ndx) = comp
		    ndx = ndx + 1
	         ENDIF
	      ENDDO

              ndx = ndx - 1
	      IF (ndx .EQ. 0) THEN 
                res(i,j,k,l,m,n) = bad_res
                CYCLE
	      ENDIF

	      IF (ndx .EQ. 1) THEN 
                res(i,j,k,l,m,n) = list(1)
                CYCLE
	      ENDIF
              CALL HEAP2_V(list, bad_com, ndx)

              median_loc = ndx/2 + 1
	      res(i,j,k,l,m,n) = list(median_loc)

	   ENDDO

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* MEDIAN ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	      IF (has_mod_void) THEN
*  ... do not allow computation across the modulo void
	         IF (k.GE.1) THEN  ! get # of modulos to curr index
	            nmod = (k-1)/modlen 
	         ELSE
	            nmod = k/modlen - 1
	         ENDIF
	         IF ( nmod .NE. nmod_prev ) THEN
	            nmod_prev = nmod
	            lo_modlim = nmod*modlen + 1
	            hi_modlim = (nmod+1) * modlen
	            lo_lim = MAX( lo_modlim, cx_lo_ss(com_cx,idim) )
	            hi_lim = MIN( hi_modlim, cx_hi_ss(com_cx,idim)  )
	         ENDIF
	      ENDIF
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      ndx = 1
	      DO ii = -hlen, hlen
* test limits of component data before using
                 IF ( k+ii .GE. lo_lim
     .          .AND. k+ii .LE. hi_lim ) THEN
                    comp = com(i,j,k+ii,l,m,n)
                 ELSE
                    comp = bad_com
                 ENDIF

	         IF ( comp .NE. bad_com ) THEN
	            list(ndx) = comp
		    ndx = ndx + 1
	         ENDIF
	      ENDDO

              ndx = ndx - 1
	      IF (ndx .EQ. 0) THEN 
                res(i,j,k,l,m,n) = bad_res
                CYCLE
	      ENDIF

	      IF (ndx .LE. 1) THEN 
                res(i,j,k,l,m,n) = list(1)
                CYCLE
	      ENDIF
              CALL HEAP2_V(list, bad_com, ndx)

              median_loc = ndx/2 + 1
	      res(i,j,k,l,m,n) = list(median_loc)

	   ENDDO
	   ENDDO

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* MEDIAN ALONG T AXIS
	ELSEIF ( idim .EQ. t_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	      IF (has_mod_void) THEN
*  ... do not allow computation across the modulo void
	         IF (l.GE.1) THEN  ! get # of modulos to curr index
	            nmod = (l-1)/modlen 
	         ELSE
	            nmod = l/modlen - 1
	         ENDIF
	         IF ( nmod .NE. nmod_prev ) THEN
	            nmod_prev = nmod
	            lo_modlim = nmod*modlen + 1
	            hi_modlim = (nmod+1) * modlen
	            lo_lim = MAX( lo_modlim, cx_lo_ss(com_cx,idim) )
	            hi_lim = MIN( hi_modlim, cx_hi_ss(com_cx,idim)  )
	         ENDIF
	      ENDIF
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      ndx = 1
	      DO ii = -hlen, hlen
* test limits of component data before using
                 IF ( l+ii .GE. lo_lim
     .          .AND. l+ii .LE. hi_lim ) THEN
                    comp = com(i,j,k,l+ii,m,n)
                 ELSE
                    comp = bad_com
                 ENDIF

	         IF ( comp .NE. bad_com ) THEN
	            list(ndx) = comp
		    ndx = ndx + 1
	         ENDIF
	      ENDDO
              ndx = ndx - 1
	      IF (ndx .EQ. 0) THEN 
                res(i,j,k,l,m,n) = bad_res
                CYCLE
	      ENDIF

	      IF (ndx .EQ. 1) THEN 
                res(i,j,k,l,m,n) = list(1)
                CYCLE
	      ENDIF
              CALL HEAP2_V(list, bad_com, ndx)

              median_loc = ndx/2 + 1
	      res(i,j,k,l,m,n) = list(median_loc)

	   ENDDO
	   ENDDO
	   ENDDO

	   ENDDO
	   ENDDO
	   ENDDO

* MEDIAN ALONG E AXIS
	ELSEIF ( idim .EQ. e_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	      IF (has_mod_void) THEN
*  ... do not allow computation across the modulo void
	         IF (m.GE.1) THEN  ! get # of modulos to curr index
	            nmod = (m-1)/modlen 
	         ELSE
	            nmod = m/modlen - 1
	         ENDIF
	         IF ( nmod .NE. nmod_prev ) THEN
	            nmod_prev = nmod
	            lo_modlim = nmod*modlen + 1
	            hi_modlim = (nmod+1) * modlen
	            lo_lim = MAX( lo_modlim, cx_lo_ss(com_cx,idim) )
	            hi_lim = MIN( hi_modlim, cx_hi_ss(com_cx,idim)  )
	         ENDIF
	      ENDIF
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      ndx = 1
	      DO ii = -hlen, hlen
* test limits of component data before using
                 IF ( m+ii .GE. lo_lim
     .          .AND. m+ii .LE. hi_lim ) THEN
                    comp = com(i,j,k,l,m+ii,n)
                 ELSE
                    comp = bad_com
                 ENDIF

	         IF ( comp .NE. bad_com ) THEN
	            list(ndx) = comp
		    ndx = ndx + 1
	         ENDIF
	      ENDDO
              ndx = ndx - 1
	      IF (ndx .EQ. 0) THEN 
                res(i,j,k,l,m,n) = bad_res
                CYCLE
	      ENDIF

	      IF (ndx .EQ. 1) THEN 
                res(i,j,k,l,m,n) = list(1)
                CYCLE
	      ENDIF
              CALL HEAP2_V(list, bad_com, ndx)

              median_loc = ndx/2 + 1
	      res(i,j,k,l,m,n) = list(median_loc)

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	   ENDDO
	   ENDDO

* MEDIAN ALONG F AXIS
	ELSEIF ( idim .EQ. f_dim ) THEN
	   DO n = lo_s6, hi_s6
	      IF (has_mod_void) THEN
*  ... do not allow computation across the modulo void
	         IF (n.GE.1) THEN  ! get # of modulos to curr index
	            nmod = (n-1)/modlen 
	         ELSE
	            nmod = n/modlen - 1
	         ENDIF
	         IF ( nmod .NE. nmod_prev ) THEN
	            nmod_prev = nmod
	            lo_modlim = nmod*modlen + 1
	            hi_modlim = (nmod+1) * modlen
	            lo_lim = MAX( lo_modlim, cx_lo_ss(com_cx,idim) )
	            hi_lim = MIN( hi_modlim, cx_hi_ss(com_cx,idim)  )
	         ENDIF
	      ENDIF
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      ndx = 1
	      DO ii = -hlen, hlen
* test limits of component data before using
                 IF ( n+ii .GE. lo_lim
     .          .AND. n+ii .LE. hi_lim ) THEN
                    comp = com(i,j,k,l,m,n+ii)
                 ELSE
                    comp = bad_com
                 ENDIF

	         IF ( comp .NE. bad_com ) THEN
	            list(ndx) = comp
		    ndx = ndx + 1
	         ENDIF
	      ENDDO
              ndx = ndx - 1
	      IF (ndx .EQ. 0) THEN 
                res(i,j,k,l,m,n) = bad_res
                CYCLE
	      ENDIF

	      IF (ndx .EQ. 1) THEN 
                res(i,j,k,l,m,n) = list(1)
                CYCLE
	      ENDIF
              CALL HEAP2_V(list, bad_com, ndx)

              median_loc = ndx/2 + 1
	      res(i,j,k,l,m,n) = list(median_loc)

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	   ENDDO

	ENDIF

	RETURN
	END
