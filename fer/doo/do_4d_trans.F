	SUBROUTINE DO_4D_TRANS( action,
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx,
     .				boxes )

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
*		have a geog Y axis that isn't actually used

* internal variable declarations:
	LOGICAL GEOG_COS_FACTOR, point, yes_ax(4), no_ax(4),
     .		at_ave, at_var, at_din, at_nbd, at_ngd, short_comp,
     .		need_cos
	INTEGER idim, i, j, ir, jr, kr, lr, ic, jc, kc, lc, grid, nbd, ngd,
     .		com_lo(4), com_hi(4), box_offset(4), offset, ind_lo, ind_hi,
     .		cos_offset
	REAL	BOX_SIZE, unit, bad_com, bad_res, box, mean, diff,
     .		xbox, ybox, zbox, tbox
	REAL*8	TM_WORLD, dsum, bsum

#ifdef unix
	include	'ferret.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'tmap_dims.parm'
	include 'xunits.cmn_text'
	external xunits_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* calling argument declarations:
	INTEGER	action, com_mr, com_cx, res_mr, res_cx
	REAL    com ( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		res ( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		boxes(*)

* equivalences
	LOGICAL  yes_xax, yes_yax, yes_zax, yes_tax,
     .		 no_xax,  no_yax,  no_zax,  no_tax

     	INTEGER	ioffset, joffset, koffset, loffset,
     .		i_lo, j_lo, k_lo, l_lo,
     .		i_hi, j_hi, k_hi, l_hi

	EQUIVALENCE ( yes_ax(1), yes_xax ),
     .		    ( yes_ax(2), yes_yax ),
     .		    ( yes_ax(3), yes_zax ),
     .		    ( yes_ax(4), yes_tax ),
     .		    (  no_ax(1),  no_xax ),
     .		    (  no_ax(2),  no_yax ),
     .		    (  no_ax(3),  no_zax ),
     .		    (  no_ax(4),  no_tax ),
     .		    ( box_offset(1), ioffset ),
     .		    ( box_offset(2), joffset ),
     .		    ( box_offset(3), koffset ),
     .		    ( box_offset(4), loffset ),
     .		    (  com_lo(1),  i_lo ),
     .		    (  com_lo(2),  j_lo ),
     .		    (  com_lo(3),  k_lo ),
     .		    (  com_lo(4),  l_lo ),
     .		    (  com_hi(1),  i_hi ),
     .		    (  com_hi(2),  j_hi ),
     .		    (  com_hi(3),  k_hi ),
     .		    (  com_hi(4),  l_hi )

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
	short_comp = at_ngd .OR. at_nbd		! for speed, only

* determine the axes to be transforomed
	DO 10 idim = 1, 4
	  yes_ax(idim) = cx_trans(idim,res_cx) .EQ. action
	  no_ax (idim) = .NOT.yes_ax(idim)
 10	CONTINUE

* get and save grid box sizes (weights) just once for efficiency
* ... save all 4 axes in one long array with offsets to find the starts
* Note that "cx_lo_ss(com_cx,idim)" is subtracted so offsets pre-compensate
*    for the axes starting above index = 1
	offset = 0
	DO 90 idim = 1, 4
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
	DO 100 idim = 1, 4
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
	    IF ( at_din ) THEN
	      unit = un_convert( line_unit_code(grid_line(idim,grid)) )
	      DO 50 i = ind_lo, ind_hi
 50	      boxes(i+offset) = boxes(i+offset) * unit
	    ENDIF
	  ENDIF
 100	CONTINUE

* impose cosine factors if latitude and longitude are involved
* 5/97 - only use COS(y) if X integration is involved and Y axis lims are VALID
	need_cos = GEOG_COS_FACTOR( y_dim,grid )
     .		.AND. yes_xax
     .	 	.AND. cx_lo_s2(com_cx) .NE. unspecified_int4
	IF ( need_cos ) THEN
	      DO 120 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
 120	      boxes(j+cos_offset)  = COS( deg2rad *       
     .           SNGL( TM_WORLD(j,grid,y_dim,box_middle)) ) 
	ELSE
	      DO 121 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
 121	      boxes(j+cos_offset)  = 1.0

	ENDIF

* do the transformation: average, integration, variance, ngood, nbad, ...

* Note: the looping structure allows all possible mixes of axis transforms
*	to be performed in a single framework
* ... set up default limits (com_lo, com_hi) in preparation for the inner loop
* "r" refers to result, "c" refers to component

* Set the range of loop indices for the component data on the transformed axes
	DO 150 idim = 1, 4
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

* LOOP OVER THE FULL RANGE OF THE RESULT
* ... also set up loop indices for non-transformed axes
*     (component loop to include only the single result point on "no" axes)
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
	    IF (at_din) THEN
               IF (com(ic,jc,kc,lc) .EQ. bad_com) THEN
                  res(ir,jr,kr,lr) = bad_res
               ELSE
                  res(ir,jr,kr,lr) = 0.0
	       ENDIF
            ELSE
               res(ir,jr,kr,lr) = bad_res
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

	    DO 210 lc = l_lo, l_hi
	    DO 210 kc = k_lo, k_hi
	    DO 210 jc = j_lo, j_hi
	    DO 210 ic = i_lo, i_hi

	      IF ( com(ic,jc,kc,lc) .EQ. bad_com ) THEN
	        nbd = nbd + 1
	      ELSE
	        ngd = ngd + 1
	      ENDIF

 210	    CONTINUE

	  ELSE

	    DO 220 lc = l_lo, l_hi
	    IF (yes_tax) tbox = boxes(lc+loffset)
	    DO 220 kc = k_lo, k_hi
	    IF (yes_zax) zbox = boxes(kc+koffset)
	    DO 220 jc = j_lo, j_hi
	    IF (yes_yax) ybox = boxes(jc+joffset)
	    DO 220 ic = i_lo, i_hi
	    IF (yes_xax) xbox = boxes(ic+ioffset)

	      IF ( com(ic,jc,kc,lc) .NE. bad_com ) THEN
	        box = xbox * ybox * zbox * tbox
	        IF (need_cos) box = box * boxes(jc+cos_offset)
	        bsum = bsum + box
	        dsum = dsum + box*com(ic,jc,kc,lc)
	      ENDIF

 220	    CONTINUE

! for variance calculation, only: a 2nd pass ...
	    IF ( at_var ) THEN
	      IF (bsum .NE. 0.0D0) THEN
	        mean = dsum / bsum
	        dsum = 0.0D0
	        DO 230 lc = l_lo, l_hi
	        IF (yes_tax) tbox = boxes(lc+loffset)
	        DO 230 kc = k_lo, k_hi
	        IF (yes_zax) zbox = boxes(kc+koffset)
	        DO 230 jc = j_lo, j_hi
	        IF (yes_yax) ybox = boxes(jc+joffset)
	        DO 230 ic = i_lo, i_hi
	        IF (yes_xax) xbox = boxes(ic+ioffset)

	          IF ( com(ic,jc,kc,lc) .NE. bad_com ) THEN
	            box = xbox * ybox * zbox * tbox
	            IF (need_cos) box = box * boxes(jc+cos_offset)
	            diff = com(ic,jc,kc,lc) - mean
	            dsum = dsum + box*diff*diff
	          ENDIF

 230	        CONTINUE
	      ENDIF
	    ENDIF

	  ENDIF

* compute result from accumulators
	  IF (at_ngd) THEN
	     res(ir,jr,kr,lr) = ngd
	  ELSEIF (at_nbd) THEN
	     res(ir,jr,kr,lr) = nbd
          ELSEIF ( bsum .EQ. 0.0D0 ) THEN
	     res(ir,jr,kr,lr) = bad_res
	  ELSEIF (at_din) THEN
	     res(ir,jr,kr,lr) = dsum
	  ELSE	
	     res(ir,jr,kr,lr) = dsum/bsum	! average or variance
	  ENDIF

 300	CONTINUE

	RETURN

	END	

