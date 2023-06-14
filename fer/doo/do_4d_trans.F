	SUBROUTINE DO_4D_TRANS( action,
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx,
     .				boxes,
     .			        weights, s_var )

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

* for a discussion of the 1-pass variance and standard deviation calculations
* see DO_VAR_SUB

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
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* V685 *acm* 7/13  Fix ticket 2083: single-y and multi-D transform
* V685+*acm* 12/13 Bug in y boxes; only when y axis in the transf.
* V695 *acm* 10/15 Fix #2319: @STD as a multi-D transform
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.
* V702 *sh* 11/16 mods for trac #2428 - calculation in gathered fragments
* V72  *acm* 6/17 Ticket 2539: Do not set res to bad-value for ave, din.
*                 Weights of zero or non-zero mark whether data is found.
* V741 *acm* 7/18 Issue 1880: bug in the box sizes changes with the v7.02 
*                 checkin. Test for the region not aligned with first or 
*                 last grid cell was incorrect.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

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
	LOGICAL GEOG_COS_FACTOR, ITSA_TRUEMONTH_AXIS,
     .		point, yes_ax(nferdims), no_ax(nferdims),
     .		at_ave, at_var, at_din, at_nbd, at_ngd, at_sum, at_std, 
     .		short_comp,need_cos, one_y, its_var_or_std, its_ave_or_din
	INTEGER idim, i, j, ir, jr, kr, lr, mr, nr, ic, jc, kc, lc, mc, nc,
     .		grid, nbd, ngd, com_lo(nferdims), com_hi(nferdims),
     .		box_offset(nferdims), offset, ind_lo, ind_hi, cos_offset
	REAL	BOX_SIZE, unit, bad_com, bad_res, box, mean, 
     .		diff,xbox, ybox, zbox, tbox, ebox, fbox,
     .		yc, yh, yh_2, bb, ybxlo, ybxhi, lo_box_lo, hi_box_hi
	REAL*8	TM_WORLD, dsum, bsum, ok,
     .		w, v, wsum_chunk, mean_chunk, mean_old, s_chunk

* calling argument declarations:
	INTEGER	action, com_mr, com_cx, res_mr, res_cx
	REAL    com     ( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .                    m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          res     ( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .                    m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
     .          weights ( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .                    m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
     .          s_var   ( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .                    m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
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
	one_y = .FALSE.

* determine the type of transform requested
	at_ave = action .EQ. trans_4d_ave
	at_var = action .EQ. trans_4d_var
	at_din = action .EQ. trans_4d_int_def
	at_nbd = action .EQ. trans_4d_bad_pt
	at_ngd = action .EQ. trans_4d_good_pt
	at_sum = action .EQ. trans_4d_sum
	at_std = action .EQ. trans_4d_std
	short_comp = at_ngd .OR. at_nbd	.OR. at_sum	! for speed, only
	its_ave_or_din = at_ave .OR. at_din
	its_var_or_std = at_var .OR. at_std

* determine the axes to be transformed
	DO idim = 1, nferdims
	  yes_ax(idim) = cx_trans(idim,res_cx) .EQ. action
	  no_ax (idim) = .NOT.yes_ax(idim)
	ENDDO

* get and save grid box sizes (weights) just once for efficiency
* ... save all 4 axes in one long array with offsets to find the starts
* Note that "cx_lo_ss(com_cx,idim)" is subtracted so offsets pre-compensate
*    for the axes starting above index = 1

      IF (.NOT. short_comp) THEN  ! boxes are not needed for ngd, nbd, sum.
	offset = 0
	DO idim = 1, nferdims
	  IF (yes_ax(idim)) THEN
	    box_offset(idim) = offset - cx_lo_ss(com_cx,idim) + 1 ! pre-comp'ed
	    offset = offset + 
     .	             cx_hi_ss(com_cx,idim)-cx_lo_ss(com_cx,idim)+1
	  ELSE
	    box_offset(idim) = -9999
	  ENDIF
	ENDDO
! COS(latitudes) also stored in the box array (8/96)
	cos_offset = offset - cx_lo_s2(com_cx) + 1

* ... save the box sizes
	DO idim = 1, nferdims
	  ind_lo = cx_lo_ss(com_cx,idim)
	  ind_hi = cx_hi_ss(com_cx,idim)
	  offset = box_offset(idim)
	  IF (yes_ax(idim)) THEN
* ... complete boxes mid-axis (incl. default full boxes at endpoints)
	   DO i = ind_lo, ind_hi
	      boxes(i+offset)  = BOX_SIZE( i, grid, idim )
	   ENDDO
* Low and high edges of end cells
	   lo_box_lo = TM_WORLD(ind_lo, grid, idim, box_lo_lim)
	   hi_box_hi = TM_WORLD(ind_hi, grid, idim, box_hi_lim)
* 11/16 re-engineered the 8/92 fix - that axis region may lie in a single box
*       in which case use the requested region limits
	   IF (ind_lo .EQ. ind_hi) THEN
	      lo_box_lo = cx_lo_ww(idim,com_cx)
	      hi_box_hi = cx_hi_ww(idim,com_cx)
	   ENDIF
* ... trim if partial box at lower end
	   IF ( lo_box_lo .LT. cx_lo_ww(idim,res_cx) ) boxes(ind_lo+offset) =
     .         boxes(ind_lo+offset) - 
     .			( cx_lo_ww(idim,res_cx) - lo_box_lo )
* ... trim if partial box at upper end
	   IF ( hi_box_hi .GT. cx_hi_ww(idim,res_cx) ) boxes(ind_hi+offset) = 
     .            boxes(ind_hi+offset) - 
     .			( hi_box_hi - cx_hi_ww(idim,res_cx) )
	   point = point .OR. boxes(ind_lo+offset).EQ.0.0
* ... perform integrations on standard units if possible
            unit = 1.
	    IF ( at_din ) THEN
	      unit = un_convert( line_unit_code(grid_line(idim,grid)) )
	      IF (ITSA_TRUEMONTH_AXIS(grid_line(idim,grid))) 
     .		  unit = un_convert(pun_day)
	      DO i = ind_lo, ind_hi
 	         boxes(i+offset) = boxes(i+offset) * unit
	      ENDDO
	    ENDIF
	  ENDIF
	ENDDO

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
	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)

C Compute the y boxes if there is a y axis; the computation may be 
C avgs in other directions but we need the ybox size at a given y
C The top and bottom boxes extend only to the coordinate locations,
C (Unless there is just one grid cell in the y direction.)

	      ybxlo = deg2rad* 
     .                TM_WORLD(j, grid, y_dim, box_lo_lim)
	      ybxhi = deg2rad* 
     .                TM_WORLD(j, grid, y_dim, box_hi_lim)
	      IF (j .EQ. cx_lo_s2(com_cx) .AND. .NOT.one_y)
     .                ybxlo = deg2rad* cx_lo_ww(y_dim,res_cx)

	      IF (j .EQ. cx_hi_s2(com_cx) .AND. .NOT.one_y)
     .                ybxhi = deg2rad* cx_hi_ww(y_dim,res_cx)

	      boxes(j+cos_offset) = ABS( SIN(ybxhi) - SIN(ybxlo) )

	      ! surface area of sphere depends only on h not on ybox
	      IF (.NOT.at_din .AND. yes_ax(y_dim)) boxes(j+box_offset(y_dim)) = 1.  

	      ! for integral need area is radius*radius* delx * h
	      ! delx already has units that include radius of earth.
	      IF ( at_din .AND. yes_ax(y_dim) ) then
 		       boxes(j+box_offset(y_dim)) =  m_radius 
	      ENDIF
	      ! Single-point Y axis or Y not part of the integral

	      IF ( at_din .AND. no_ax(y_dim))
     .           boxes(j+cos_offset) = COS( deg2rad * 
     .             TM_WORLD(j,grid,y_dim,box_middle) )
	   ENDDO
	ELSE
	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	      boxes(j+cos_offset)  = 1.0
	   ENDDO
	ENDIF

      ENDIF  ! skipped computing boxes for short_comp

* do the transformation: average, integration, variance, ngood, nbad, ...

* Note: the looping structure allows all possible mixes of axis transforms
*	to be performed in a single framework
* ... set up default limits (com_lo, com_hi) in preparation for the inner loop
* "r" refers to result, "c" refers to component

* Set the range of loop indices for the component data on the transformed axes
	DO idim = 1, nferdims
	  IF (yes_ax(idim)) THEN
	    com_lo(idim) = cx_lo_ss(com_cx,idim)
	    com_hi(idim) = cx_hi_ss(com_cx,idim)
	  ENDIF
	ENDDO

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
	DO nr = cx_lo_s6(res_cx), cx_hi_s6(res_cx)
	IF (no_fax) THEN
	   n_lo = nr
	   n_hi = nr
	ENDIF
	DO mr = cx_lo_s5(res_cx), cx_hi_s5(res_cx)
	IF (no_eax) THEN
	   m_lo = mr
	   m_hi = mr
	ENDIF
	DO lr = cx_lo_s4(res_cx), cx_hi_s4(res_cx)
	IF (no_tax) THEN
	   l_lo = lr
	   l_hi = lr
	ENDIF
	DO kr = cx_lo_s3(res_cx), cx_hi_s3(res_cx)
	IF (no_zax) THEN
	  k_lo = kr
	  k_hi = kr
	ENDIF
	DO jr = cx_lo_s2(res_cx), cx_hi_s2(res_cx)
	IF (no_yax) THEN
	  j_lo = jr
	  j_hi = jr
	ENDIF
	DO ir = cx_lo_s1(res_cx), cx_hi_s1(res_cx)
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

	    IF (at_din .AND. com(ic,jc,kc,lc,mc,nc) .NE. bad_com) 
     .        res(ir,jr,kr,lr,mr,nr) = 0.0
            CYCLE  ! next iteration of full-range loops
          ENDIF

* initialize the inner loop accumulators
	  IF (its_var_or_std) THEN
* ... initializations were done from IS_TRANS
* ... start from the last accumulated value to support split/gather
	    wsum_chunk = weights(ir,jr,kr,lr,mr,nr)
	    mean_chunk = res    (ir,jr,kr,lr,mr,nr)
	    s_chunk    = s_var  (ir,jr,kr,lr,mr,nr)
	  ELSE
	    dsum = 0.0D0	! data
	    bsum = 0.0D0	! boxes
	    ok	 = 0.0D0	! count valid
	    ngd  = 0
	    nbd  = 0
	  ENDIF

* now loop over the component data accumulating results along relevant axes
* ... two separate loops for efficiency, only -- keep @ngd and @nbd fast


	  IF (short_comp) THEN

	    DO nc = n_lo, n_hi
	    DO mc = m_lo, m_hi
	    DO lc = l_lo, l_hi
	    DO kc = k_lo, k_hi
	    DO jc = j_lo, j_hi
	    DO ic = i_lo, i_hi

	      IF ( com(ic,jc,kc,lc,mc,nc) .EQ. bad_com ) THEN
	        nbd = nbd + 1
	      ELSE
	        ngd = ngd + 1
	        IF (at_sum) THEN
	           dsum = dsum + com(ic,jc,kc,lc,mc,nc)
	           ok = ok + 1
	        ENDIF
	      ENDIF

	    ENDDO
	    ENDDO
	    ENDDO
	    ENDDO
	    ENDDO
	    ENDDO

	  ELSE

	    DO nc = n_lo, n_hi
	    IF (yes_fax) fbox = boxes(nc+noffset)
	    DO mc = m_lo, m_hi
	    IF (yes_eax) ebox = boxes(mc+moffset)
	    DO lc = l_lo, l_hi
	    IF (yes_tax) tbox = boxes(lc+loffset)
	    DO kc = k_lo, k_hi
	    IF (yes_zax) zbox = boxes(kc+koffset)
	    DO jc = j_lo, j_hi
	    IF (yes_yax) ybox = boxes(jc+joffset)
	    DO ic = i_lo, i_hi
	    IF (yes_xax) xbox = boxes(ic+ioffset)

	      IF ( com(ic,jc,kc,lc,mc,nc) .NE. bad_com ) THEN
	        box = xbox * ybox * zbox * tbox * ebox * fbox
	        IF (need_cos) box = box * boxes(jc+cos_offset)
	      	IF (its_var_or_std) THEN
	          v = com(ic,jc,kc,lc,mc,nc)
	          w = box
	          wsum_chunk = wsum_chunk + w
	          mean_old = mean_chunk
	          mean_chunk = mean_old + (w/wsum_chunk)*(v-mean_old)
	          s_chunk = s_chunk + w * (v-mean_old)*(v-mean_chunk)
	        ELSE
	          bsum = bsum + box
	          dsum = dsum + box*com(ic,jc,kc,lc,mc,nc)
	        ENDIF
	      ENDIF

	    ENDDO
	    ENDDO
	    ENDDO
	    ENDDO
	    ENDDO
	    ENDDO

	  ENDIF

* *sh* 11/16 -- accumulate values, only in this routine.  Finalize elsewhere
* acm notes: 
*   o at_ngd and at_nbd, add the current result to res(ir,jr,... )?  
*   o If bsum EQ 0, do not set the accumulating result to bad_res as
*     we did in the non-scatter and finalize code.  The weights take 
*     care of knowing if any good data was found.

	  IF (at_ngd) THEN
	     res(ir,jr,kr,lr,mr,nr) = ngd  
	  ELSEIF (at_nbd) THEN
	     res(ir,jr,kr,lr,mr,nr) = nbd  
	  ELSEIF (at_sum .AND. ngd.GT.0) THEN
	     res(ir,jr,kr,lr,mr,nr) = dsum
	     weights(ir,jr,kr,lr,mr,nr) = weights(ir,jr,kr,lr,mr,nr) + ok ! weights are #valid
	  ELSEIF (its_ave_or_din) THEN
	      res    (ir,jr,kr,lr,mr,nr) = res    (ir,jr,kr,lr,mr,nr) + dsum
	      weights(ir,jr,kr,lr,mr,nr) = weights(ir,jr,kr,lr,mr,nr) + bsum
	  ELSEIF (its_var_or_std) THEN
	    res    (ir,jr,kr,lr,mr,nr) = mean_chunk
	    weights(ir,jr,kr,lr,mr,nr) = wsum_chunk
	    s_var  (ir,jr,kr,lr,mr,nr) = s_chunk
	  
	     CONTINUE
	  ENDIF

	ENDDO  ! Loops over full range of result
	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO

	RETURN

	END

