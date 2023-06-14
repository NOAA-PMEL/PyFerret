	SUBROUTINE DO_4D_WGT( action,
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

* Return weights as for avg/integral on the grid

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA 
* 7/2014

* V692 7/14 *acm* new transform @WGT, ticket 2180
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.
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
     .		point, yes_ax(nferdims), no_ax(nferdims),need_cos, one_y
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
	one_y = .FALSE.

* determine the axes to be transformed
	DO idim = 1, nferdims
	  yes_ax(idim) = cx_trans(idim,res_cx) .EQ. action
	  no_ax (idim) = .NOT.yes_ax(idim)
	ENDDO

* get and save grid box sizes (weights) just once for efficiency
* ... save all axes in one long array with offsets to find the starts
* Note that "cx_lo_ss(com_cx,idim)" is subtracted so offsets pre-compensate
*    for the axes starting above index = 1

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
            point = .FALSE.
	    IF ( ind_lo .EQ. ind_hi ) THEN    ! 8/92
              boxes(ind_lo+offset) = cx_hi_ww(idim,res_cx)
     .				 - cx_lo_ww(idim,res_cx)
              point = boxes(ind_lo+offset) .EQ. 0.0
            ELSE
* ... partial box at lower end
	      boxes(ind_lo+offset) = TM_WORLD(ind_lo, grid, idim, box_hi_lim)
     .				 - cx_lo_ww(idim,res_cx)

* ... complete boxes mid-axis
	      DO i = ind_lo+1, ind_hi-1
	         boxes(i+offset)  = BOX_SIZE( i, grid, idim )
 	      ENDDO
* ... partial box at upper end
	      boxes(ind_hi+offset) = cx_hi_ww(idim,res_cx)
     .		   - TM_WORLD(ind_hi, grid, idim, box_lo_lim)
            ENDIF
* ... perform integrations on standard units if possible
            unit = 1.
	    unit = un_convert( line_unit_code(grid_line(idim,grid)) )
	    IF (ITSA_TRUEMONTH_AXIS(grid_line(idim,grid))) 
     .		  unit = un_convert(pun_day)

	    DO i = ind_lo, ind_hi
 	      boxes(i+offset) = boxes(i+offset) * unit
	    ENDDO
 	    IF (point) boxes(ind_lo+offset) = 1.
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
	      
	      yh = ybxhi-ybxlo  
	      boxes(j+cos_offset) = ABS( SIN(ybxhi) - SIN(ybxlo) )/yh

	   ENDDO
	ELSE
	   DO j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	      boxes(j+cos_offset)  = 1.0
	   ENDDO
	ENDIF

* do the transformation: return box size 

* Set default box size for untransformed axes
* (multiplying  by 1.0 to have nil effect)
	xbox = 1.0
	ybox = 1.0
	zbox = 1.0
	tbox = 1.0
	ebox = 1.0
	fbox = 1.0

* Loop over the component data, returning box sizes

	n_lo = cx_lo_s6(res_cx)
	n_hi = cx_hi_s6(res_cx)
	m_lo = cx_lo_s5(res_cx)
	m_hi = cx_hi_s5(res_cx)
	l_lo = cx_lo_s4(res_cx)
	l_hi = cx_hi_s4(res_cx)
	k_lo = cx_lo_s3(res_cx)
	k_hi = cx_hi_s3(res_cx)
	j_lo = cx_lo_s2(res_cx)
	j_hi = cx_hi_s2(res_cx)
	i_lo = cx_lo_s1(res_cx)
	i_hi = cx_hi_s1(res_cx)

* Set the range of loop indices for the component data on the transformed axes
	DO idim = 1, nferdims
	  IF (yes_ax(idim)) THEN
	    com_lo(idim) = cx_lo_ss(com_cx,idim)
	    com_hi(idim) = cx_hi_ss(com_cx,idim)
	  ENDIF
	ENDDO
	
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

	    box = xbox * ybox * zbox * tbox * ebox * fbox
	    IF (need_cos) box = box * boxes(jc+cos_offset)
	    res(ic,jc,kc,lc,mc,nc) = box

	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO

	RETURN

	END

