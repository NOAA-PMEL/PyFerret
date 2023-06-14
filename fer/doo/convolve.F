	SUBROUTINE CONVOLVE(	idim, hlen, wt,
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx )

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
* convolve the component grid, com, with the weight function, wt
* along axis idim
* note: the component context may not be of adequate size for the full
*	calculation.  Missing data flags will be inserted where computation is
*	impossible
* also: when bad data points are encountered in the component data all
*	result data depending on it are flagged as bad, too

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/2/89
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN 
* V500: bug fix *sh* & *kob* 4/99 -  add code to make sure that component
*                                    limits are adequate
* V550: support for subspan modulo axes -- ensure that convolution does not
*	cross the void point
*
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* V720: added support for convolutions on DSG ragged arrays
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include	'xdset_info.cmn_text'

* calling argument declarations:
	INTEGER	idim, hlen, com_mr, com_cx, res_mr, res_cx
	REAL	wt(-hlen:hlen),
     .		com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .		     m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .		res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .		     m2lot:m2hit,m2loe:m2hie,m2lof:m2hif )

* internal variable declarations:
	LOGICAL TM_ITS_SUBSPAN_MODULO, TM_ITSA_DSG, has_mod_void
	INTEGER	TM_MODULO_LINE_DIM, CGRID_AXIS, 
     .		TM_DSG_DSET_FROM_GRID, TM_DSG_NFEATURES,
     .		i, j, k, l, m, n, ii,
     .		lo_sub(nferdims), hi_sub(nferdims), lo_lim, hi_lim,
     .		hlen1, lo_modlim, hi_modlim, nmod, nmod_prev, modlen,
     .		grid, dset, nfeatures, orientation, mr_list(2)
	REAL	bad_com, bad_res, comp, sum

* internal (convenience) equivalences
	INTEGER lo_s1, lo_s2, lo_s3, lo_s4, lo_s5, lo_s6,
     .          hi_s1, hi_s2, hi_s3, hi_s4, hi_s5, hi_s6
	EQUIVALENCE (lo_s1, lo_sub(1)) , (lo_s2, lo_sub(2)),
     .		    (lo_s3, lo_sub(3)) , (lo_s4, lo_sub(4)),
     .		    (lo_s5, lo_sub(5)) , (lo_s6, lo_sub(6)),
     .		    (hi_s1, hi_sub(1)) , (hi_s2, hi_sub(2)),
     .		    (hi_s3, hi_sub(3)) , (hi_s4, hi_sub(4)),
     .		    (hi_s5, hi_sub(5)) , (hi_s6, hi_sub(6))


* DSGs are handled as a separate case
* The calls to NON_ARRAY_SUBSC and DSG_ARRAY_SUBSC handle things
* so that the result context will be correct

	grid = cx_grid(com_cx)
	IF ( TM_ITSA_DSG(grid) ) THEN
	   dset = TM_DSG_DSET_FROM_GRID(grid)
	   nfeatures = TM_DSG_NFEATURES(grid)

	   orientation = dsg_orientation(dset)
	   mr_list(1) = com_mr
	   mr_list(2) = res_mr
	   CALL NON_ARRAY_SUBSC( mr_list, 2 )
	   CALL DSG_ARRAY_SUBSC( mr_list, 2, orientation )

           CALL CONVOLVE_DSG (idim, hlen, wt, dset, nfeatures,
     .                       com, com_mr, com_cx,
     .                       res, res_mr, res_cx )
	   RETURN
	ENDIF

* initialize
	bad_com = mr_bad_data( res_mr )
	bad_res = mr_bad_data( com_mr )
	hlen1 = hlen
	lo_lim = cx_lo_ss(com_cx,idim)
	hi_lim = cx_hi_ss(com_cx,idim) 
	nmod_prev = -99999

* compute limits within which calculation is possible
	DO ii = 1, nferdims
	   lo_sub(ii) = cx_lo_ss(res_cx,ii)
	   hi_sub(ii) = cx_hi_ss(res_cx,ii)
	ENDDO

* it his a subspan modulo axis?
	has_mod_void = TM_ITS_SUBSPAN_MODULO( CGRID_AXIS(idim,com_cx) )
	IF (has_mod_void) THEN
	   modlen = TM_MODULO_LINE_DIM( CGRID_AXIS(idim,com_cx) )
	   has_mod_void = lo_lim.LE.0 .OR. hi_lim.GE.modlen
	ENDIF

* !temp - not optimized for speed yet
* ... though treating each axis separately is to avoid testing inside the loop
* CONVOLVE ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
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
	      sum = 0.0
	      DO ii = -hlen1, hlen1
* test limits of component data before using
                 IF ( i+ii .GE. lo_lim
     .          .AND. i+ii .LE. hi_lim ) THEN
                    comp = com(i+ii,j,k,l,m,n)
                 ELSE
                    res(i,j,k,l,m,n) = bad_res
                    GOTO 110
                 ENDIF

	         IF ( comp .EQ. bad_com ) THEN
	            res(i,j,k,l,m,n) = bad_res
	            GOTO 110
	         ELSE
	            sum = sum + comp*wt(ii)
	         ENDIF
	      ENDDO
	      res(i,j,k,l,m,n) = sum

 110	   CONTINUE
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

* CONVOLVE ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
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
	   DO i = lo_s1, hi_s1
	      sum = 0.0
	      DO ii = -hlen1, hlen1
* test limits of component data before using
                 IF ( j+ii .GE. lo_lim
     .          .AND. j+ii .LE. hi_lim ) THEN
                    comp = com(i,j+ii,k,l,m,n)
                 ELSE
                    res(i,j,k,l,m,n) = bad_res
                    GOTO 210
                 ENDIF

	         IF ( comp .EQ. bad_com ) THEN
	            res(i,j,k,l,m,n) = bad_res
	            GOTO 210
	         ELSE
	            sum = sum + comp*wt(ii)
	         ENDIF
	      ENDDO
	      res(i,j,k,l,m,n) = sum

 210	   CONTINUE
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

* CONVOLVE ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
	   DO k = lo_s3, hi_s3
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
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      sum = 0.0
	      DO ii = -hlen1, hlen1
* test limits of component data before using
                 IF ( k+ii .GE. lo_lim
     .          .AND. k+ii .LE. hi_lim ) THEN
                    comp = com(i,j,k+ii,l,m,n)
                 ELSE
                    res(i,j,k,l,m,n) = bad_res
                    GOTO 310
                 ENDIF

	         IF ( comp .EQ. bad_com ) THEN
	            res(i,j,k,l,m,n) = bad_res
	            GOTO 310
	         ELSE
	            sum = sum + comp*wt(ii)
	         ENDIF
	      ENDDO
	      res(i,j,k,l,m,n) = sum

 310	   CONTINUE
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

* CONVOLVE ALONG T AXIS
	ELSE IF ( idim .EQ. t_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	   DO l = lo_s4, hi_s4
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
	   DO k = lo_s3, hi_s3
	   DO j = lo_s2, hi_s2
	   DO i = lo_s1, hi_s1
	      sum = 0.0
	      DO ii = -hlen1, hlen1
* test limits of component data before using
                 IF ( l+ii .GE. lo_lim
     .          .AND. l+ii .LE. hi_lim ) THEN
                    comp = com(i,j,k,l+ii,m,n)
                 ELSE
                    res(i,j,k,l,m,n) = bad_res
                    GOTO 410
                 ENDIF

	         IF ( comp .EQ. bad_com ) THEN
	            res(i,j,k,l,m,n) = bad_res
	            GOTO 410
	         ELSE
	            sum = sum + comp*wt(ii)
	         ENDIF
	      ENDDO
	      res(i,j,k,l,m,n) = sum

 410	   CONTINUE
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

* CONVOLVE ALONG E AXIS
	ELSE IF ( idim .EQ. e_dim ) THEN
	   DO n = lo_s6, hi_s6
	   DO m = lo_s5, hi_s5
	      IF (has_mod_void) THEN
*  ... do not allow convolution across the modulo void
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
	      sum = 0.0
	      DO ii = -hlen1, hlen1
* test limits of component data before using
                 IF ( m+ii .GE. lo_lim
     .          .AND. m+ii .LE. hi_lim ) THEN
                    comp = com(i,j,k,l,m+ii,n)
                 ELSE
                    res(i,j,k,l,m,n) = bad_res
                    GOTO 510
                 ENDIF

	         IF ( comp .EQ. bad_com ) THEN
	            res(i,j,k,l,m,n) = bad_res
	            GOTO 510
	         ELSE
	            sum = sum + comp*wt(ii)
	         ENDIF
	      ENDDO
	      res(i,j,k,l,m,n) = sum

 510	   CONTINUE
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

* CONVOLVE ALONG F AXIS
	ELSE IF ( idim .EQ. f_dim ) THEN
	   DO n = lo_s6, hi_s6
	      IF (has_mod_void) THEN
*  ... do not allow convolution across the modulo void
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
	      sum = 0.0
	      DO ii = -hlen1, hlen1
* test limits of component data before using
                 IF ( n+ii .GE. lo_lim
     .          .AND. n+ii .LE. hi_lim ) THEN
                    comp = com(i,j,k,l,m,n+ii)
                 ELSE
                    res(i,j,k,l,m,n) = bad_res
                    GOTO 610
                 ENDIF

	         IF ( comp .EQ. bad_com ) THEN
	            res(i,j,k,l,m,n) = bad_res
	            GOTO 610
	         ELSE
	            sum = sum + comp*wt(ii)
	         ENDIF
	      ENDDO
	      res(i,j,k,l,m,n) = sum
 610	   CONTINUE
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

	ENDIF

	RETURN
	END
