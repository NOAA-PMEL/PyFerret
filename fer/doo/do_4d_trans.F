	SUBROUTINE DO_4D_TRANS( action,
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx,
     .				boxes )

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
* perform a multi-axis transformation

* multi-axis transformations are those special cases where sequential
* application of single-axis transforms may yield improper results. For
* example, averaging within a circular region would give inappropriate
* results if carried out sequentially: the first axis of averaging would
* be correct but the second axis would weight each averaged row from the
* first axis equally, despite the fact that they represented different
* numbers of data points.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V430: 7/1/96 - based on DO_XY_AVE_INT
* ... comments from DO_XY_AVE_INT:
* V200:  6/5/89
*       10/11/89 - modified array declarations using XMEM_SUBSC.CMN
* V230: 8/10/92  - results were wrong for integration within a single box
* V430: 7/30/96 - IBM port - change yes/no_[xyzt]ax declaration from
*	          INTEGER to LOGICAL *kob*
* V440: 8/23/96 - added cos_factor, was not applying COS(lat) to XZ calcs
* V450: 5/97 -  only apply COS(y) if X integration is involved and Y axis
*		limits are valid (abstract expressions like Z[G=temp] can
*		have a geog Y axis that is not actually used
* 3/06 - apply Andrew*s suggested correction for XY integ., bug1348
* 6/06 - Fix to Andrew*s correction for XY integ., bug1348
* 4/10 - Fix #1735; For averages other than in Y direction the above fix 
*        failed to set ybox sizes and cos correction. Still need to set these.
* 8/10 - One further fix to #1735
* 2/11 - One further fix to #1348: @DIN over 1 grid cell in Y
* V68  *acm* 1/12  ifdef double_p for double-precision ferret.
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xunits.cmn_text'
	external xunits_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data

* internal variable declarations:
	LOGICAL GEOG_COS_FACTOR, point, yes_ax(nferdims), no_ax(nferdims),
     .		at_ave, at_var, at_din, at_nbd, at_ngd, at_sum, short_comp,
     .		need_cos, one_y
	INTEGER idim, i, j, ir, jr, kr, lr, mr, nr, ic, jc, kc, lc, mc, nc,
     .		grid, nbd, ngd, com_lo(nferdims), com_hi(nferdims),
     .		box_offset(nferdims), offset, ind_lo, ind_hi, cos_offset
	REAL	BOX_SIZE, unit, bad_com, bad_res, box, mean, diff,
     .		xbox, ybox, zbox, tbox, ebox, fbox,
     .		yc, yh, yh_2, bb, ybxlo, ybxhi
	REAL*8	TM_WORLD, dsum, bsum

* calling argument declarations:
	INTEGER	action, com_mr, com_cx, res_mr, res_cx
	REAL    com ( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .                m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          res ( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .                m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
     .		boxes(*)

* equivalences
	LOGICAL yes_xax, yes_yax, yes_zax,
     .          yes_tax, yes_eax, yes_fax,
     .          no_xax,  no_yax,  no_zax,
     .          no_tax,  no_eax,  no_fax

     	INTEGER	ioffset, joffset, koffset, loffset, moffset, noffset,
     .		i_lo, j_lo, k_lo, l_lo, m_lo, n_lo,
     .		i_hi, j_hi, k_hi, l_hi, m_hi, n_hi

	EQUIVALENCE ( yes_ax(1), yes_xax ),
     .		    ( yes_ax(2), yes_yax ),
     .		    ( yes_ax(3), yes_zax ),
     .		    ( yes_ax(4), yes_tax ),
     .		    ( yes_ax(5), yes_eax ),
     .		    ( yes_ax(6), yes_fax ),
     .		    (  no_ax(1),  no_xax ),
     .		    (  no_ax(2),  no_yax ),
     .		    (  no_ax(3),  no_zax ),
     .		    (  no_ax(4),  no_tax ),
     .		    (  no_ax(5),  no_eax ),
     .		    (  no_ax(6),  no_fax ),
     .		    ( box_offset(1), ioffset ),
     .		    ( box_offset(2), joffset ),
     .		    ( box_offset(3), koffset ),
     .		    ( box_offset(4), loffset ),
     .		    ( box_offset(5), moffset ),
     .		    ( box_offset(6), noffset ),
     .		    (  com_lo(1),  i_lo ),
     .		    (  com_lo(2),  j_lo ),
     .		    (  com_lo(3),  k_lo ),
     .		    (  com_lo(4),  l_lo ),
     .		    (  com_lo(5),  m_lo ),
     .		    (  com_lo(6),  n_lo ),
     .		    (  com_hi(1),  i_hi ),
     .		    (  com_hi(2),  j_hi ),
     .		    (  com_hi(3),  k_hi ),
     .		    (  com_hi(4),  l_hi ),
     .		    (  com_hi(5),  m_hi ),
     .		    (  com_hi(6),  n_hi )

* --- end of introductory code ---

* initialize
	grid = mr_grid( com_mr )
	bad_com = mr_bad_data( com_mr )
	bad_res = mr_bad_data( res_mr )
        point = .FALSE.

* determine the type of transform requested
	at_ave = action .EQ. trans_4d_ave
	at_var = action .EQ. trans_4d_var
	at_din = action .EQ. trans_4d_int_def
	at_nbd = action .EQ. trans_4d_bad_pt
	at_ngd = action .EQ. trans_4d_good_pt
	at_sum = action .EQ. trans_4d_sum
	short_comp = at_ngd .OR. at_nbd	.OR. at_sum	! for speed, only

* determine the axes to be transformed
	DO 10 idim = 1, nferdims
	  yes_ax(idim) = cx_trans(idim,res_cx) .EQ. action
	  no_ax (idim) = .NOT.yes_ax(idim)
 10	CONTINUE

* get and save grid box sizes (weights) just once for efficiency
* ... save all 4 axes in one long array with offsets to find the starts
* Note that "cx_lo_ss(com_cx,idim)" is subtracted so offsets pre-compensate
*    for the axes starting above index = 1

      IF (.NOT. short_comp) THEN  ! boxes are not needed for ngd, nbd, sum.
	offset = 0
	DO 90 idim = 1, nferdims
	  IF (yes_ax(idim)) THEN
	    box_offset(idim) = offset - cx_lo_ss(com_cx,idim) + 1 ! pre-comp'ed
	    offset = offset + 
     .	             cx_hi_ss(com_cx,idim)-cx_lo_ss(com_cx,idim)+1
	  ELSE
	    box_offset(idim) = -9999
	  ENDIF
 90	CONTINUE
! COS(latitudes) also stored in the box array (8/96)
	cos_offset = offset - cx_lo_s2(com_cx) + 1

* ... save the box sizes
	DO 100 idim = 1, nferdims
	  ind_lo = cx_lo_ss(com_cx,idim)
	  ind_hi = cx_hi_ss(com_cx,idim)
	  offset = box_offset(idim)
	  IF (yes_ax(idim)) THEN
            IF ( ind_lo .EQ. ind_hi ) THEN    ! 8/92
              boxes(ind_lo+offset) = cx_hi_ww(idim,res_cx)
     .				 - cx_lo_ww(idim,res_cx)
              point = point  .OR.   boxes(ind_lo+offset) .EQ. 0.0
            ELSE
* ... partial box at lower end
	      boxes(ind_lo+offset) = TM_WORLD(ind_lo, grid, idim, box_hi_lim)
     .				 - cx_lo_ww(idim,res_cx)

* ... complete boxes mid-axis
	      DO 40 i = ind_lo+1, ind_hi-1
 40	      boxes(i+offset)  = BOX_SIZE( i, grid, idim )
* ... partial box at upper end
	      boxes(ind_hi+offset) = cx_hi_ww(idim,res_cx)
     .		   - TM_WORLD(ind_hi, grid, idim, box_lo_lim)
            ENDIF
* ... perform integrations on standard units if possible
            unit = 1.
	    IF ( at_din ) THEN
	      unit = un_convert( line_unit_code(grid_line(idim,grid)) )
	      DO 50 i = ind_lo, ind_hi
 	      boxes(i+offset) = boxes(i+offset) * unit
 50           continue
	    ENDIF
	  ENDIF
 100	CONTINUE

* impose cosine factors if latitude and longitude are involved
* 5/97 - only use COS(y) if X integration is involved and Y axis lims are VALID
* 3/06 - apply Andrew*s suggested correction for XY integ., bug1348
* 6/06   fix the correction; boxes(i+offset) is in meters, so convert back to 
*        degrees, then radians. Compute the factor, and then convert to meters.
C See this thread:
C http://www.pmel.noaa.gov/maillists/tmap/ferret_users/fu_2009/msg00506.html
C And also ticket 1348. The surface area of sphere depends only on h
C http://mathworld.wolfram.com/Zone.html. Fix for @DIN too.

	need_cos = GEOG_COS_FACTOR( y_dim,grid )
     .		.AND. yes_xax
     .	 	.AND. cx_lo_s2(com_cx) .NE. unspecified_int4
        IF ( at_sum ) need_cos = .FALSE.
        one_y = (cx_lo_ww(y_dim,res_cx) .EQ. cx_hi_ww(y_dim,res_cx))

	IF ( need_cos ) THEN
	   DO 120 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)

C Compute the y boxes if there is a y axis; the computation may be 
C avgs in other directions but we need the ybox size at a given y
C The top and bottom boxes extend only to the coordinate locations,
C (Unless there is just one grid cell in the y direction.)

#ifdef double_p
	      ybxlo = deg2rad* 
     .                TM_WORLD(j, grid, y_dim, box_lo_lim)
	      ybxhi = deg2rad* 
     .                TM_WORLD(j, grid, y_dim, box_hi_lim)
#else
	      ybxlo = deg2rad* 
     .                SNGL( TM_WORLD(j, grid, y_dim, box_lo_lim) )
	      ybxhi = deg2rad* 
     .                SNGL( TM_WORLD(j, grid, y_dim, box_hi_lim) )
#endif
	      IF (j .EQ. cx_lo_s2(com_cx) .AND. .NOT.one_y)
     .                ybxlo = deg2rad* cx_lo_ww(y_dim,res_cx)

	      IF (j .EQ. cx_hi_s2(com_cx) .AND. .NOT.one_y)
     .                ybxhi = deg2rad* cx_hi_ww(y_dim,res_cx)

	      boxes(j+cos_offset) = ABS( SIN(ybxhi) - SIN(ybxlo) )

	      ! surface area of sphere depends only on h not on ybox
	      IF (.NOT.at_din) boxes(j+box_offset(y_dim)) = 1.  

	      ! for integral need area is radius*radius* delx * h
	      ! delx already has units that include radius of earth.
	      IF ( at_din ) then
 		       boxes(j+box_offset(y_dim)) =  m_radius 
	      ENDIF
	      ! Single-point Y axis or Y not part of the integral

#ifdef double_p
	      IF ( at_din .AND. no_ax(y_dim))
     .           boxes(j+cos_offset) = COS( deg2rad * 
     .             TM_WORLD(j,grid,y_dim,box_middle) )
#else
	      IF ( at_din .AND. no_ax(y_dim))
     .           boxes(j+cos_offset) = COS( deg2rad * 
     .           SNGL( TM_WORLD(j,grid,y_dim,box_middle) ) )
#endif
 120       CONTINUE
	ELSE
	      DO 121 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
 121	      boxes(j+cos_offset)  = 1.0
	ENDIF

      ENDIF  ! skipped computing boxes for short_comp

* do the transformation: average, integration, variance, ngood, nbad, ...

* Note: the looping structure allows all possible mixes of axis transforms
*	to be performed in a single framework
* ... set up default limits (com_lo, com_hi) in preparation for the inner loop
* "r" refers to result, "c" refers to component

* Set the range of loop indices for the component data on the transformed axes
	DO 150 idim = 1, nferdims
	  IF (yes_ax(idim)) THEN
	    com_lo(idim) = cx_lo_ss(com_cx,idim)
	    com_hi(idim) = cx_hi_ss(com_cx,idim)
	  ENDIF
 150	CONTINUE

* Set default box size for untransformed axes
* (multiplying  by 1.0 to have nil effect)
	xbox = 1.0
	ybox = 1.0
	zbox = 1.0
	tbox = 1.0
	ebox = 1.0
	fbox = 1.0

* LOOP OVER THE FULL RANGE OF THE RESULT
* ... also set up loop indices for non-transformed axes
*     (component loop to include only the single result point on "no" axes)
	DO 300 nr = cx_lo_s6(res_cx), cx_hi_s6(res_cx)
	IF (no_fax) THEN
	   n_lo = nr
	   n_hi = nr
	ENDIF
	DO 300 mr = cx_lo_s5(res_cx), cx_hi_s5(res_cx)
	IF (no_eax) THEN
	   m_lo = mr
	   m_hi = mr
	ENDIF
	DO 300 lr = cx_lo_s4(res_cx), cx_hi_s4(res_cx)
	IF (no_tax) THEN
	   l_lo = lr
	   l_hi = lr
	ENDIF
	DO 300 kr = cx_lo_s3(res_cx), cx_hi_s3(res_cx)
	IF (no_zax) THEN
	  k_lo = kr
	  k_hi = kr
	ENDIF
	DO 300 jr = cx_lo_s2(res_cx), cx_hi_s2(res_cx)
	IF (no_yax) THEN
	  j_lo = jr
	  j_hi = jr
	ENDIF
	DO 300 ir = cx_lo_s1(res_cx), cx_hi_s1(res_cx)
	IF (no_xax) THEN
	  i_lo = ir
	  i_hi = ir
	ENDIF

* pathological case: world coordinate range of zero along an xform axis

          IF ( point ) THEN    ! 8/92
            nc = n_lo
            mc = m_lo
            lc = l_lo
            kc = k_lo
            jc = j_lo
            ic = i_lo

	    IF (at_din) THEN
               IF (com(ic,jc,kc,lc,mc,nc) .EQ. bad_com) THEN
                  res(ir,jr,kr,lr,mr,nr) = bad_res
               ELSE
                  res(ir,jr,kr,lr,mr,nr) = 0.0
	       ENDIF
            ELSE
               res(ir,jr,kr,lr,mr,nr) = bad_res
            ENDIF
            GOTO 300
          ENDIF

* initialize the inner loop accumulators
	  dsum = 0.0D0	! data
	  bsum = 0.0D0	! boxes
	  ngd  = 0
	  nbd  = 0

* now loop over the component data accumulating results along relevant axes
* ... two separate loops for efficiency, only -- keep @ngd and @nbd fast


	  IF (short_comp) THEN

	    DO 210 nc = n_lo, n_hi
	    DO 210 mc = m_lo, m_hi
	    DO 210 lc = l_lo, l_hi
	    DO 210 kc = k_lo, k_hi
	    DO 210 jc = j_lo, j_hi
	    DO 210 ic = i_lo, i_hi

	      IF ( com(ic,jc,kc,lc,mc,nc) .EQ. bad_com ) THEN
	        nbd = nbd + 1
	      ELSE
	        ngd = ngd + 1
		IF (at_sum) dsum = dsum + com(ic,jc,kc,lc,mc,nc)
	      ENDIF

 210	    CONTINUE

	  ELSE

	    DO 220 nc = n_lo, n_hi
	    IF (yes_fax) fbox = boxes(nc+noffset)
	    DO 220 mc = m_lo, m_hi
	    IF (yes_eax) ebox = boxes(mc+moffset)
	    DO 220 lc = l_lo, l_hi
	    IF (yes_tax) tbox = boxes(lc+loffset)
	    DO 220 kc = k_lo, k_hi
	    IF (yes_zax) zbox = boxes(kc+koffset)
	    DO 220 jc = j_lo, j_hi
	    IF (yes_yax) ybox = boxes(jc+joffset)
	    DO 220 ic = i_lo, i_hi
	    IF (yes_xax) xbox = boxes(ic+ioffset)

	      IF ( com(ic,jc,kc,lc,mc,nc) .NE. bad_com ) THEN
	        box = xbox * ybox * zbox * tbox * ebox * fbox
	        IF (need_cos) box = box * boxes(jc+cos_offset)
	        bsum = bsum + box
	        dsum = dsum + box*com(ic,jc,kc,lc,mc,nc)
	      ENDIF

 220	    CONTINUE

! for variance calculation, only: a 2nd pass ...
	    IF ( at_var ) THEN
	      IF (bsum .NE. 0.0D0) THEN
	        mean = dsum / bsum
	        dsum = 0.0D0
	        DO 230 nc = n_lo, n_hi
	        IF (yes_fax) fbox = boxes(nc+noffset)
	        DO 230 mc = m_lo, m_hi
	        IF (yes_eax) ebox = boxes(mc+moffset)
	        DO 230 lc = l_lo, l_hi
	        IF (yes_tax) tbox = boxes(lc+loffset)
	        DO 230 kc = k_lo, k_hi
	        IF (yes_zax) zbox = boxes(kc+koffset)
	        DO 230 jc = j_lo, j_hi
	        IF (yes_yax) ybox = boxes(jc+joffset)
	        DO 230 ic = i_lo, i_hi
	        IF (yes_xax) xbox = boxes(ic+ioffset)

	          IF ( com(ic,jc,kc,lc,mc,nc) .NE. bad_com ) THEN
	            box = xbox * ybox * zbox * tbox * ebox * fbox
	            IF (need_cos) box = box * boxes(jc+cos_offset)
	            diff = com(ic,jc,kc,lc,mc,nc) - mean
	            dsum = dsum + box*diff*diff
	          ENDIF

 230	        CONTINUE
	      ENDIF
	    ENDIF

	  ENDIF

* compute result from accumulators
	  IF (at_ngd) THEN
	     res(ir,jr,kr,lr,mr,nr) = ngd
	  ELSEIF (at_nbd) THEN
	     res(ir,jr,kr,lr,mr,nr) = nbd
	  ELSEIF (at_sum .AND. ngd.GT.0) THEN
	     res(ir,jr,kr,lr,mr,nr) = dsum
          ELSEIF ( bsum .EQ. 0.0D0 ) THEN
	     res(ir,jr,kr,lr,mr,nr) = bad_res
	  ELSEIF (at_din) THEN
	     res(ir,jr,kr,lr,mr,nr) = dsum
	  ELSE
	     res(ir,jr,kr,lr,mr,nr) = dsum/bsum	! average or variance
	  ENDIF

 300	CONTINUE

	RETURN

	END

