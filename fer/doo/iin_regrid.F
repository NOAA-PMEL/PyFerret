	SUBROUTINE IIN_REGRID (	cx_lims, idim, dst_lo, dst_hi,
     .				src, msrc, dst, mdst, box, itot, status )

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
*
* integrate the component along axis idim producing an "indefinite" integral
* placing the result on a grid which is 1 cell longer than the source grid,
* as described in ticket #2346.

* The indefinite integral transform returns a result that is shifted a half-cell 
* down from where it should be. Because the full cell is being integrated, the 
* result really ought to sit at the upper edge of the cell, not the cell center.

* 	programmer - Ansley Manke
* 	NOAA/PMEL,Seattle,WA - 
* 	revision 1.0 - 4/2017

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xcontext.cmn'
	include 'xunits.cmn_text'
	external xunits_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data

* CALLING ARGUMENT DECLARATIONS:
	INTEGER cx_lims, idim, dst_lo, dst_hi, msrc, mdst, itot, status
	REAL    src( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          dst( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
     .		box(mr_lo_ss(msrc,idim):mr_hi_ss(msrc,idim))

* internal variable declarations:
	LOGICAL	GEOG_COS_FACTOR, ITSA_TRUEMONTH_AXIS, TM_DFPEQ,
     .          need_cos, blank
	INTEGER i, j, k, l, m, n, sgrid, dgrid, axis, src_lo, src_hi,
     .          ii, jj, kk, ll, mm, nn, is, js, ks, ls, ms, ns
	REAL	BOX_SIZE, unit, bad_src, bad_dst, cos_factor,
     .          ybxlo, ybxhi, bx
	REAL*8	TM_WORLD, dsum
	

* source and destination grids
	sgrid = mr_grid( msrc )
	dgrid = mr_grid( mdst )
	axis = grid_line( idim, sgrid )

* LIMITS FOR CALCULATION
        src_lo = mr_lo_ss(msrc,idim)
        src_hi = mr_hi_ss(msrc,idim)

* FLAG FOR BAD/MISSING DATA
        bad_src = mr_bad_data(msrc)
        bad_dst = mr_bad_data(mdst)

* Get and save grid box sizes (weights) just once for efficiency
* Check that grid boxes are the same size on source and destination axes

        is = src_lo
        DO 100 i = dst_lo, dst_hi
           IF (is.LT.src_lo .OR. is.GT.src_hi) GOTO 100
           bx =  BOX_SIZE( i, dgrid, idim )
           box(is)  = BOX_SIZE( i, sgrid, idim )
           IF (.NOT.TM_DFPEQ( bx, box(is) ) ) GOTO 5100
           is = is + 1
 100	CONTINUE

* No overlap between grids; return all-missing result
	blank = src_lo.GT.dst_hi .OR. src_hi.LT.dst_lo

* perform integrations on standard units if possible
	unit = un_convert( line_unit_code(axis) )
	IF (ITSA_TRUEMONTH_AXIS(axis)) unit = un_convert(pun_day)
	DO 110 i = src_lo, src_hi
 110	box(i) = box(i) * unit

* INTEGRATE
* ALONG X AXIS
* (note: X integration is a special case because it requires COSINE(lat) )
	IF ( idim .EQ. x_dim ) THEN
	   need_cos = GEOG_COS_FACTOR( idim, sgrid )
     .                .AND. cx_lo_s2(cx_lims).NE.unspecified_int4
	   DO 210 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	   cos_factor = 1.D0
	   IF ( need_cos ) THEN
#ifdef double_p
	      cos_factor = COS( deg2rad *       
     .            TM_WORLD(j, sgrid, y_dim, box_middle) ) 
              ybxlo = deg2rad* 
     .            TM_WORLD(j,sgrid,y_dim,box_lo_lim)
              ybxhi = deg2rad* 
     .            TM_WORLD(j,sgrid,y_dim,box_hi_lim)
#else
	      cos_factor = COS( deg2rad *       
     .            SNGL( TM_WORLD(j, sgrid, y_dim, box_middle) ) ) 
              ybxlo = deg2rad* 
     .            SNGL( TM_WORLD(j,sgrid,y_dim,box_lo_lim) )
              ybxhi = deg2rad* 
     .            SNGL( TM_WORLD(j,sgrid,y_dim,box_hi_lim) )
#endif
              bx = ABS(ybxhi - ybxlo)
	      cos_factor = ABS( SIN(ybxhi) - SIN(ybxlo) )/ bx
	   ENDIF
	   DO 210 n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
	   DO 210 m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
	   DO 210 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	   DO 210 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	      dsum = 0.0D0
	      is = src_lo
	      DO 205 i = dst_lo, dst_hi

	         dst(i,j,k,l,m,n) = bad_dst
		 IF (blank) GOTO 205
	         IF (is.GT.src_hi .AND. is.EQ.dst_hi) GOTO 205

	         dst(i,j,k,l,m,n) = dsum * cos_factor

		 IF ( src(is,j,k,l,m,n) .EQ. bad_src ) THEN
* ... reject all data after a bad point
	            DO 200 ii = i, dst_hi
 200	              dst(ii,j,k,l,m,n) = bad_dst
	            GOTO 210
	         ENDIF
	         dsum = dsum + box(is)*src(is,j,k,l,m,n)
		 is = is + 1
 205	      CONTINUE
	      IF (is .EQ. dst_hi) dst(is,j,k,l,m,n) = dsum * cos_factor
 210	   CONTINUE

* ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO 310 n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
	   DO 310 m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
	   DO 310 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	   DO 310 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	   DO 310 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
	      dsum = 0.0D0
	      js = src_lo
	      DO 305 j = dst_lo, dst_hi

	         dst(i,j,k,l,m,n) = bad_dst
		 IF (blank) GOTO 305
	         IF (js.GT.src_hi .AND. js.EQ.dst_hi) GOTO 305

	         dst(i,j,k,l,m,n) = dsum
	         
		 IF ( src(i,js,k,l,m,n) .EQ. bad_src ) THEN
* ... reject all data after a bad point
	            DO 300 jj = j, dst_hi
 300	            dst(i,jj,k,l,m,n) = bad_dst
	            GOTO 310
	         ENDIF
	         dsum = dsum + box(js)*src(i,js,k,l,m,n)
		 js = js + 1
 305	      CONTINUE
	      IF (js .EQ. dst_hi) dst(i,js,k,l,m,n) = dsum
 310	   CONTINUE


* ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 410 n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
	   DO 410 m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
	   DO 410 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	   DO 410 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	   DO 410 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
	      dsum = 0.0D0
	      ks = src_lo
	      DO 405 k = dst_lo, dst_hi

	         dst(i,j,k,l,m,n) = bad_dst
		 IF (blank) GOTO 405
	         IF (ks.GT.src_hi .AND. ks.EQ.dst_hi) GOTO 405

	         dst(i,j,k,l,m,n) = dsum

	         IF ( src(i,j,ks,l,m,n) .EQ. bad_src ) THEN
* ... reject all data after a bad point
	            DO 400 kk = k, dst_hi
 400	            dst(i,j,kk,l,m,n) = bad_dst
	            GOTO 410
	         ENDIF
	         dsum = dsum + box(ks)*src(i,j,ks,l,m,n)
		 ks = ks + 1
 405	      CONTINUE
	      IF (ks .EQ. dst_hi) dst(i,j,ks,l,m,n) = dsum
 410	   CONTINUE

* ALONG T AXIS
	ELSEIF ( idim .EQ. t_dim ) THEN
	   DO 510 n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
	   DO 510 m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
	   DO 510 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	   DO 510 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	   DO 510 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
	      dsum = 0.0D0
	      ls = src_lo
	      DO 505 l = dst_lo, dst_hi
	      
	         dst(i,j,k,l,m,n) = bad_dst
		 IF (blank) GOTO 505
	         IF (ls.GT.src_hi .AND. ls.EQ.dst_hi) GOTO 505

	         dst(i,j,k,l,m,n) = dsum

		 IF (ls .GT. src_hi) GOTO 510
	         IF ( src(i,j,k,ls,m,n) .EQ. bad_src ) THEN
* ... reject all data after a bad point
	            DO 500 ll = l, dst_hi
 500	            dst(i,j,k,ll,m,n) = bad_dst
	            GOTO 510
	         ENDIF
	         dsum = dsum + box(ls)*src(i,j,k,ls,m,n)
		 ls = ls + 1
 505	      CONTINUE
	      IF (ls .EQ. dst_hi) dst(i,j,k,ls,m,n) = dsum
 510	   CONTINUE

* ALONG E AXIS
	ELSEIF ( idim .EQ. e_dim ) THEN
	   DO 610 n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
	   DO 610 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	   DO 610 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	   DO 610 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	   DO 610 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
	      dsum = 0.0D0
	      ms = src_lo
	      DO 605 m = dst_lo, dst_hi
	      
	         dst(i,j,k,l,m,n) = bad_dst
		 IF (blank) GOTO 605
	         IF (ms.GT.src_hi .AND. ms.EQ.dst_hi) GOTO 605

	         dst(i,j,k,l,m,n) = dsum

	         IF ( src(i,j,k,l,ms,n) .EQ. bad_src ) THEN
* ... reject all data after a bad point
	            DO 600 mm = m, dst_hi
 600	            dst(i,j,k,l,mm,n) = bad_dst
	            GOTO 610
	         ENDIF
	         dsum = dsum + box(ms)*src(i,j,k,l,ms,n)
		 ms = ms + 1
 605	      CONTINUE
	      IF (ms .EQ. dst_hi) dst(i,j,k,l,ms,n) = dsum
 610	   CONTINUE

* ALONG F AXIS
	ELSEIF ( idim .EQ. f_dim ) THEN
	   DO 710 m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
	   DO 710 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	   DO 710 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	   DO 710 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	   DO 710 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
	      dsum = 0.0D0
	      ns = src_lo
	      DO 705 n = dst_lo, dst_hi
	      
	         dst(i,j,k,l,m,n) = bad_dst
		 IF (blank) GOTO 705
	         IF (ns.GT.src_hi .AND. ns.EQ.dst_hi) GOTO 705

	         dst(i,j,k,l,m,n) = dsum

	         IF ( src(i,j,k,l,m,ns) .EQ. bad_src ) THEN
* ... reject all data after a bad point
	            DO 700 nn = n, dst_hi
 700	            dst(i,j,k,l,m,nn) = bad_dst
	            GOTO 710
	         ENDIF
	         dsum = dsum + box(ns)*src(i,j,k,l,m,ns)
		 ns = ns + 1
 705	      CONTINUE
	      IF (ns .EQ. dst_hi) dst(i,j,k,l,m,ns) = dsum
 710	   CONTINUE

	ENDIF

 5000	RETURN

 5100	CALL ERRMSG(ferr_regrid, status,
     .    '@IIN regridding must be to axis with same grid-cell sizes', *5000 )


 5200	CALL ERRMSG(ferr_regrid, status,
     .    '@IIN regridding on non-overlapping grids', *5000 )

	END
