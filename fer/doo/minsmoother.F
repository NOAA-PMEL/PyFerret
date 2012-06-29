	SUBROUTINE MINSMOOTHER(	idim, wlen, 
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
* Compute the minimum of the requested length, along axis idim
* note: the component context may not be of adequate size for the full
*	calculation.  Missing data flags will be inserted where computation is
*	impossible. If there is any data present in the window, the minimum
*       is computed based on that data. Minima are computed with the data
*       that is available in partial windows at the edges of the domain. 

* programmer - ansley manke based on convolve.
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)

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
     .		modlen
	REAL    bad_com, bad_res, comp, cmin

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
	IF ( MOD(wlen,2) .EQ. 0)  hlen = (wlen+1)/2

* compute limits within which calculation is possible
	DO 10 ii = 1, nferdims
	   lo_sub(ii) = cx_lo_ss(res_cx,ii)
	   hi_sub(ii) = cx_hi_ss(res_cx,ii)
 10	CONTINUE

* it this a subspan modulo axis?
	has_mod_void = TM_ITS_SUBSPAN_MODULO( CGRID_AXIS(idim,com_cx) )
	IF (has_mod_void) THEN
	   modlen = TM_MODULO_LINE_DIM( CGRID_AXIS(idim,com_cx) )
	   has_mod_void = lo_lim.LE.0 .OR. hi_lim.GE.modlen
	ENDIF

* ... though treating each axis separately is to avoid testing inside the loop
* SMOOTH ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO 110 n = lo_s6, hi_s6
	   DO 110 m = lo_s5, hi_s5
	   DO 110 l = lo_s4, hi_s4
	   DO 110 k = lo_s3, hi_s3
	   DO 110 j = lo_s2, hi_s2
	   DO 110 i = lo_s1, hi_s1
	      IF (has_mod_void) THEN
*  ... do not allow smoothing across the modulo void
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
	      ndx = 0
	      cmin = ABS(bad_com)
	      DO 100 ii = -hlen, hlen
* test limits of component data before using
                 IF ( i+ii .GE. lo_lim
     .          .AND. i+ii .LE. hi_lim ) THEN
                    comp = com(i+ii,j,k,l,m,n)
                 ELSE
                    comp = bad_com
                 ENDIF

	         IF ( comp .NE. bad_com ) THEN
		    cmin = MIN(cmin, comp)
		    ndx = ndx + 1
	         ENDIF
 100	      CONTINUE

	      IF (ndx .EQ. 0) THEN 
                res(i,j,k,l,m,n) = bad_res
                GOTO 110
	      ELSE
                res(i,j,k,l,m,n) = cmin
	      ENDIF

 110	   CONTINUE

* SMOOTH ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO 210 n = lo_s6, hi_s6
	   DO 210 m = lo_s5, hi_s5
	   DO 210 l = lo_s4, hi_s4
	   DO 210 k = lo_s3, hi_s3
	   DO 210 j = lo_s2, hi_s2
	      IF (has_mod_void) THEN
*  ... do not allow smoothing across the modulo void
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
	   DO 210 i = lo_s1, hi_s1
	      ndx = 0
	      cmin = ABS(bad_com)

	      DO 200 ii = -hlen, hlen
* test limits of component data before using
                 IF ( j+ii .GE. lo_lim
     .          .AND. j+ii .LE. hi_lim ) THEN
                    comp = com(i,j+ii,k,l,m,n)
                 ELSE
                    comp = bad_com
                 ENDIF

	         IF ( comp .NE. bad_com ) THEN
	            cmin = MIN(cmin, comp)
		    ndx = ndx + 1
	         ENDIF
 200	      CONTINUE

	      IF (ndx .EQ. 0) THEN 
                res(i,j,k,l,m,n) = bad_res
                GOTO 210
	      ELSE 
                res(i,j,k,l,m,n) = cmin
	      ENDIF

 210	   CONTINUE

* SMOOTH ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 310 n = lo_s6, hi_s6
	   DO 310 m = lo_s5, hi_s5
	   DO 310 l = lo_s4, hi_s4
	   DO 310 k = lo_s3, hi_s3
	      IF (has_mod_void) THEN
*  ... do not allow smoothing across the modulo void
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

	   DO 310 j = lo_s2, hi_s2
	   DO 310 i = lo_s1, hi_s1
	      ndx = 0
	      cmin = ABS(bad_com)
	      DO 300 ii = -hlen, hlen
* test limits of component data before using
                 IF ( k+ii .GE. lo_lim
     .          .AND. k+ii .LE. hi_lim ) THEN
                    comp = com(i,j,k+ii,l,m,n)
                 ELSE
                    comp = bad_com
                 ENDIF

	         IF ( comp .NE. bad_com ) THEN
	            cmin = MIN(cmin, comp)
		    ndx = ndx + 1
	         ENDIF
 300	      CONTINUE

	      IF (ndx .EQ. 0) THEN 
                res(i,j,k,l,m,n) = bad_res
                GOTO 310
	      ELSE
                res(i,j,k,l,m,n) = cmin
	      ENDIF
 310	   CONTINUE

* SMOOTH ALONG T AXIS
	ELSEIF ( idim .EQ. t_dim ) THEN
	   DO 410 n = lo_s6, hi_s6
	   DO 410 m = lo_s5, hi_s5
	   DO 410 l = lo_s4, hi_s4
	      IF (has_mod_void) THEN
*  ... do not allow smoothing across the modulo void
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
	   DO 410 k = lo_s3, hi_s3
	   DO 410 j = lo_s2, hi_s2
	   DO 410 i = lo_s1, hi_s1
	      ndx = 0
	      cmin = ABS(bad_com)
	      DO 400 ii = -hlen, hlen
* test limits of component data before using
                 IF ( l+ii .GE. lo_lim
     .          .AND. l+ii .LE. hi_lim ) THEN
                    comp = com(i,j,k,l+ii,m,n)
                 ELSE
                    comp = bad_com
                 ENDIF

	         IF ( comp .NE. bad_com ) THEN
	            cmin = MIN(cmin, comp)
		    ndx = ndx + 1
	         ENDIF
 400	      CONTINUE

	      IF (ndx .EQ. 0) THEN 
                res(i,j,k,l,m,n) = bad_res
                GOTO 410
	      ELSE 
                res(i,j,k,l,m,n) = cmin
	      ENDIF
 410	   CONTINUE

* SMOOTH ALONG E AXIS
	ELSEIF ( idim .EQ. e_dim ) THEN
	   DO 510 n = lo_s6, hi_s6
	   DO 510 m = lo_s5, hi_s5
	      IF (has_mod_void) THEN
*  ... do not allow smoothing across the modulo void
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
	   DO 510 l = lo_s4, hi_s4
	   DO 510 k = lo_s3, hi_s3
	   DO 510 j = lo_s2, hi_s2
	   DO 510 i = lo_s1, hi_s1
	      ndx = 0
	      cmin = ABS(bad_com)
	      DO 500 ii = -hlen, hlen
* test limits of component data before using
                 IF ( m+ii .GE. lo_lim
     .          .AND. m+ii .LE. hi_lim ) THEN
                    comp = com(i,j,k,l,m+ii,n)
                 ELSE
                    comp = bad_com
                 ENDIF

	         IF ( comp .NE. bad_com ) THEN
	            cmin = MIN(cmin, comp)
		    ndx = ndx + 1
	         ENDIF
 500	      CONTINUE

	      IF (ndx .EQ. 0) THEN 
                res(i,j,k,l,m,n) = bad_res
                GOTO 510
	      ELSE 
                res(i,j,k,l,m,n) = cmin
	      ENDIF
 510	   CONTINUE

* SMOOTH ALONG F AXIS
	ELSEIF ( idim .EQ. f_dim ) THEN
	   DO 610 n = lo_s6, hi_s6
	      IF (has_mod_void) THEN
*  ... do not allow smoothing across the modulo void
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
	   DO 610 m = lo_s5, hi_s5
	   DO 610 l = lo_s4, hi_s4
	   DO 610 k = lo_s3, hi_s3
	   DO 610 j = lo_s2, hi_s2
	   DO 610 i = lo_s1, hi_s1
	      ndx = 0
	      cmin = ABS(bad_com)
	      DO 600 ii = -hlen, hlen
* test limits of component data before using
                 IF ( n+ii .GE. lo_lim
     .          .AND. n+ii .LE. hi_lim ) THEN
                    comp = com(i,j,k,l,m,n+ii)
                 ELSE
                    comp = bad_com
                 ENDIF

	         IF ( comp .NE. bad_com ) THEN
	            cmin = MIN(cmin, comp)
		    ndx = ndx + 1
	         ENDIF
 600	      CONTINUE

	      IF (ndx .EQ. 0) THEN 
                res(i,j,k,l,m,n) = bad_res
                GOTO 610
	      ELSE 
                res(i,j,k,l,m,n) = cmin
	      ENDIF
 610	   CONTINUE

	ENDIF

	RETURN
	END
