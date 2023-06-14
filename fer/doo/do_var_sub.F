	SUBROUTINE DO_VAR_SUB(	idim, com_lo, com_hi,
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx,
     .				box, stddev,
     .				wsum, s )

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
* determine the variance along the indicated axis
* nested with DO_VARIANCE in order to ease the dimension checking of box

* COSINE weighting not applied because this is a strictly 1D calculation
* see DO_4D_TRANS for the contrasting multi-dimensional calculation

* calculation is done in a single pass (a change 11/2016)
* using the algorithm of West (1979) as described succintly in Wikipedia
* as the "Weighted incremental algorithm" for computing variance
* see D. H. D. West (1979). Communications of the ACM, 22, 9, 532-535:
*     Updating Mean and Variance Estimates: An Improved Method
* Note that the new algorithm may treat the unequal weghting of points
* differently than the old.  The new is better to be trusted IMHO.

****
* The algorithm:
*
* These initializations already done from IS_TRANS
*    wSum = 0
*    mean = 0    ! "res" in this code
*    S =0
*
* This routine performs this loop (see Wikipedia reference)
* v,w  is the value,weight pair contributed by each grid cell
*        wSum = wSum + w
*        meanOld = mean
*        mean = meanOld + (w / wSum) * (v - meanOld)
*        S = S + w * (v - meanOld) * (v - mean)
*	
* Split/gather memory management may cause this routine to be called many times
* in the finalize step we will compute
*    variance = S / wSum
****

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V230:   3/2/92
* V420: 11/95 - converted diff and mean to DP so variance is calc in DP
*	 2/96 - fix bug introduced in 11/95: CGRID_AXIS has wrong args
* V664:  8/10 - implement robust method for computing variances
* V672+: 1/12 - previous checkin had wrong indices for result. Return to 
*               divide by n, not n+1
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
*
* v687  3/14  *acm* Option to return standard deviation
* v702 *sh* 11/16 -  mods for trac #2428 - calculation in gathered fragments
* v73+ *acm*12/17 -  issue #1849, if box size 0 skip it - resulted in missing
*                    result where result should be computed
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'

* internal variable declarations:
	LOGICAL stddev
	INTEGER CGRID_AXIS, i, j, k, l, m, n,
     .          grid, pt, lo, hi, axis
	REAL	BOX_SIZE, bad_com, bad_res
	REAL*8	TM_WORLD, dsum, bsum
	REAL*8  w, v, wsum_chunk, mean_chunk, mean_old, s_chunk

* calling argument declarations:
	INTEGER	idim, com_lo, com_hi, com_mr, com_cx, res_mr, res_cx
	REAL    com ( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .                m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          res ( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .                m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
     .          wsum( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .                m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
     .          s   ( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .                m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
     .		box( com_lo:com_hi )

* --- end of introductory code ---

* initialize
	grid = mr_grid( com_mr )
	axis = CGRID_AXIS( idim, com_cx )
	pt   = cx_lo_ss( res_cx, idim )
	lo   = com_lo
	hi   = com_hi
	bad_com = mr_bad_data( com_mr )
	bad_res = mr_bad_data( res_mr )

* get and save grid box sizes (weights) just once for efficiency
	DO i = lo, hi
	   box(i)  = BOX_SIZE( i, grid, idim )
	ENDDO
* 11/16 ... end boxes may be only partial cells
	IF  (TM_WORLD(lo, grid, idim, box_lo_lim)
     .  .LT. cx_lo_ww(idim,res_cx)                ) box(lo) = 
     .			TM_WORLD(lo, grid, idim, box_hi_lim)
     .		      - cx_lo_ww(idim,res_cx)
	IF  (TM_WORLD(hi, grid, idim, box_hi_lim)
     .  .GT. cx_hi_ww(idim,res_cx)                ) box(hi) =
     .			cx_hi_ww(idim,res_cx)
     .		      - TM_WORLD(hi, grid, idim, box_lo_lim)

!	box(lo) = TM_WORLD(lo, grid, idim, box_hi_lim)
!     .		- cx_lo_ww(idim,res_cx)
!	box(hi) = cx_hi_ww(idim,res_cx)
!     .		- TM_WORLD(hi, grid, idim, box_lo_lim)

* ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	      wsum_chunk = wsum(pt,j,k,l,m,n)
	      mean_chunk = res (pt,j,k,l,m,n)
	      s_chunk    = s   (pt,j,k,l,m,n)
	      DO i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	         w = box(i)
	         IF ( w .EQ. 0. ) CYCLE
	         v = com(i,j,k,l,m,n)
	         IF ( v .EQ. bad_com ) CYCLE
	         wsum_chunk = wsum_chunk + w
	         mean_old = mean_chunk
	         mean_chunk = mean_old + (w/wsum_chunk)*(v-mean_old)
	         s_chunk = s_chunk + w * (v-mean_old)*(v-mean_chunk)
	      ENDDO
	      wsum(pt,j,k,l,m,n) = wsum_chunk
	      res (pt,j,k,l,m,n) = mean_chunk
	      s   (pt,j,k,l,m,n) = s_chunk
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
	      wsum_chunk = wsum(i,pt,k,l,m,n)
	      mean_chunk = res (i,pt,k,l,m,n)
	      s_chunk    = s   (i,pt,k,l,m,n)
	      DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	         w = box(j)
	         IF ( w .EQ. 0. ) CYCLE
	         v = com(i,j,k,l,m,n)
	         IF ( v .EQ. bad_com ) CYCLE
	         wsum_chunk = wsum_chunk + w
	         mean_old = mean_chunk
	         mean_chunk = mean_old + (w/wsum_chunk)*(v-mean_old)
	         s_chunk = s_chunk + w * (v-mean_old)*(v-mean_chunk)
	      ENDDO
	      wsum(i,pt,k,l,m,n) = wsum_chunk
	      res (i,pt,k,l,m,n) = mean_chunk
	      s   (i,pt,k,l,m,n) = s_chunk
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
	      wsum_chunk = wsum(i,j,pt,l,m,n)
	      mean_chunk = res (i,j,pt,l,m,n)
	      s_chunk    = s   (i,j,pt,l,m,n)
	      DO k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	         w = box(k)
	         IF ( w .EQ. 0. ) CYCLE
	         v = com(i,j,k,l,m,n)
	         IF ( v .EQ. bad_com ) CYCLE
	         wsum_chunk = wsum_chunk + w
	         mean_old = mean_chunk
	         mean_chunk = mean_old + (w/wsum_chunk)*(v-mean_old)
	         s_chunk = s_chunk + w * (v-mean_old)*(v-mean_chunk)
	      ENDDO
	      wsum(i,j,pt,l,m,n) = wsum_chunk
	      res (i,j,pt,l,m,n) = mean_chunk
	      s   (i,j,pt,l,m,n) = s_chunk
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

	      wsum_chunk = wsum(i,j,k,pt,m,n)
	      mean_chunk = res (i,j,k,pt,m,n)
	      s_chunk    = s   (i,j,k,pt,m,n)
	      DO l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	         w = box(l)
	         IF ( w .EQ. 0. ) CYCLE
	         v = com(i,j,k,l,m,n)
	         IF ( v .EQ. bad_com ) CYCLE
	         wsum_chunk = wsum_chunk + w
	         mean_old = mean_chunk
	         mean_chunk = mean_old + (w/wsum_chunk)*(v-mean_old)
	         s_chunk = s_chunk + w * (v-mean_old)*(v-mean_chunk)
	      ENDDO
	      wsum(i,j,k,pt,m,n) = wsum_chunk
	      res (i,j,k,pt,m,n) = mean_chunk
	      s   (i,j,k,pt,m,n) = s_chunk
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
	      wsum_chunk = wsum(i,j,k,l,pt,n)
	      mean_chunk = res (i,j,k,l,pt,n)
	      s_chunk    = s   (i,j,k,l,pt,n)
	      DO m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	         w = box(m)
	         IF ( w .EQ. 0. ) CYCLE
	         v = com(i,j,k,l,m,n)
	         IF ( v .EQ. bad_com ) CYCLE
	         wsum_chunk = wsum_chunk + w
	         mean_old = mean_chunk
	         mean_chunk = mean_old + (w/wsum_chunk)*(v-mean_old)
	         s_chunk = s_chunk + w * (v-mean_old)*(v-mean_chunk)
	      ENDDO
	      wsum(i,j,k,l,pt,n) = wsum_chunk
	      res (i,j,k,l,pt,n) = mean_chunk
	      s   (i,j,k,l,pt,n) = s_chunk
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
	      wsum_chunk = wsum(i,j,k,l,m,pt)
	      mean_chunk = res (i,j,k,l,m,pt)
	      s_chunk    = s   (i,j,k,l,m,pt)
	      DO n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	         w = box(n)
	         IF ( w .EQ. 0. ) CYCLE
	         v = com(i,j,k,l,m,n)
	         IF ( v .EQ. bad_com ) CYCLE
	         wsum_chunk = wsum_chunk + w
	         mean_old = mean_chunk
	         mean_chunk = mean_old + (w/wsum_chunk)*(v-mean_old)
	         s_chunk = s_chunk + w * (v-mean_old)*(v-mean_chunk)
	      ENDDO
	      wsum(i,j,k,l,m,pt) = wsum_chunk
	      res (i,j,k,l,m,pt) = mean_chunk
	      s   (i,j,k,l,m,pt) = s_chunk
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	ENDIF

* success
	RETURN
	END
