	SUBROUTINE MINSMOOTHER(	idim, wlen, 
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx, 
     .				list )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
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
	include	'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'

* calling argument declarations:
	INTEGER	idim, wlen, com_mr, com_cx, res_mr, res_cx
	REAL	com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .          list(*)

* internal variable declarations:
	LOGICAL TM_ITS_SUBSPAN_MODULO, has_mod_void
	INTEGER	TM_MODULO_LINE_DIM, CGRID_AXIS, 
     .		i, j, k, l, ii, lo_sub(4), hi_sub(4), lo_lim, hi_lim,
     .		hlen, lo_modlim, hi_modlim, nmod, nmod_prev, 
     .		modlen
	REAL    bad_com, bad_res, comp, cmin

* internal (convenience) equivalences
	INTEGER	lo_s1, lo_s2, lo_s3, lo_s4, hi_s1, hi_s2, hi_s3, hi_s4
	EQUIVALENCE (lo_s1, lo_sub(1)) , (lo_s2, lo_sub(2)),
     .		    (lo_s3, lo_sub(3)) , (lo_s4, lo_sub(4)),
     .		    (hi_s1, hi_sub(1)) , (hi_s2, hi_sub(2)),
     .		    (hi_s3, hi_sub(3)) , (hi_s4, hi_sub(4))

* initialize
	bad_com = mr_bad_data( res_mr )
	bad_res = mr_bad_data( com_mr )
	hlen1 = hlen
	lo_lim = cx_lo_ss(com_cx,idim)
	hi_lim = cx_hi_ss(com_cx,idim) 
	nmod_prev = -99999

	hlen = wlen/2
	IF ( MOD(wlen,2) .EQ. 0)  hlen = (wlen+1)/2

* compute limits within which calculation is possible
	DO 10 ii = 1, 4
	   lo_sub(ii) = cx_lo_ss(res_cx,ii)
	   hi_sub(ii) = cx_hi_ss(res_cx,ii)
 10	CONTINUE

* it this a subspan modulo axis?
	has_mod_void = TM_ITS_SUBSPAN_MODULO( CGRID_AXIS(idim,com_cx) )
	IF (has_mod_void) THEN
	   modlen = TM_MODULO_LINE_DIM( CGRID_AXIS(idim,com_cx) )
	   has_mod_void = lo_lim.LE.0 .OR. hi_lim.GE.modlen
	ENDIF

* !temp - not optimized for speed yet
* ... though treating each axis separately is to avoid testing inside the loop
* CONVOLVE ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO 110 l = lo_s4, hi_s4
	   DO 110 k = lo_s3, hi_s3
	   DO 110 j = lo_s2, hi_s2
	   DO 110 i = lo_s1, hi_s1
	      IF (has_mod_void) THEN
*  ... do not allow convolution across the modulo void
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
	      cmin = ABS(bad_com)
	      DO 100 ii = -hlen, hlen
* test limits of component data before using
                 IF ( i+ii .GE. lo_lim
     .          .AND. i+ii .LE. hi_lim ) THEN
                    comp = com(i+ii,j,k,l)
                 ELSE
                    comp = bad_com
                 ENDIF

	         IF ( comp .NE. bad_com ) THEN
		    cmin = MIN(cmin, comp)
		    ndx = ndx + 1
	         ENDIF
 100	      CONTINUE

              ndx = ndx - 1
	      IF (ndx .EQ. 0) THEN 
                res(i,j,k,l) = bad_res
                GOTO 110
	      ELSE
                res(i,j,k,l) = cmin
	      ENDIF

 110	   CONTINUE

* CONVOLVE ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO 210 l = lo_s4, hi_s4
	   DO 210 k = lo_s3, hi_s3
	   DO 210 j = lo_s2, hi_s2
	      IF (has_mod_void) THEN
*  ... do not allow convolution across the modulo void
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
	      ndx = 1
	      cmin = ABS(bad_com)

	      DO 200 ii = -hlen, hlen
* test limits of component data before using
                 IF ( j+ii .GE. lo_lim
     .          .AND. j+ii .LE. hi_lim ) THEN
                    comp = com(i,j+ii,k,l)
                 ELSE
                    comp = bad_com
                 ENDIF

	         IF ( comp .NE. bad_com ) THEN
	            cmin = MIN(cmin, comp)
		    ndx = ndx + 1
	         ENDIF
 200	      CONTINUE

              ndx = ndx - 1
	      IF (ndx .EQ. 0) THEN 
                res(i,j,k,l) = bad_res
                GOTO 210
	      ELSE 
                res(i,j,k,l) = cmin
	      ENDIF

 210	   CONTINUE

* CONVOLVE ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 310 l = lo_s4, hi_s4
	   DO 310 k = lo_s3, hi_s3
	      IF (has_mod_void) THEN
*  ... do not allow convolution across the modulo void
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
	      ndx = 1
	      cmin = ABS(bad_com)
	      DO 300 ii = -hlen, hlen
* test limits of component data before using
                 IF ( k+ii .GE. lo_lim
     .          .AND. k+ii .LE. hi_lim ) THEN
                    comp = com(i,j,k+ii,l)
                 ELSE
                    comp = bad_com
                 ENDIF

	         IF ( comp .NE. bad_com ) THEN
	            cmin = MIN(cmin, comp)
		    ndx = ndx + 1
	         ENDIF
 300	      CONTINUE

              ndx = ndx - 1
	      IF (ndx .EQ. 0) THEN 
                res(i,j,k,l) = bad_res
                GOTO 310
	      ELSE
                res(i,j,k,l) = cmin
	      ENDIF
 310	   CONTINUE

* CONVOLVE ALONG T AXIS
	ELSE
	   DO 410 l = lo_s4, hi_s4
	      IF (has_mod_void) THEN
*  ... do not allow convolution across the modulo void
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
	      ndx = 1
	      cmin = ABS(bad_com)
	      DO 400 ii = -hlen, hlen
* test limits of component data before using
                 IF ( l+ii .GE. lo_lim
     .          .AND. l+ii .LE. hi_lim ) THEN
                    comp = com(i,j,k,l+ii)
                 ELSE
                    comp = bad_com
                 ENDIF

	         IF ( comp .NE. bad_com ) THEN
	            cmin = MIN(cmin, comp)
		    ndx = ndx + 1
	         ENDIF
 400	      CONTINUE
              ndx = ndx - 1
	      IF (ndx .EQ. 0) THEN 
                res(i,j,k,l) = bad_res
                GOTO 410
	      ELSE 
                res(i,j,k,l) = cmin
	      ENDIF
 410	   CONTINUE
	ENDIF

	RETURN
	END	
