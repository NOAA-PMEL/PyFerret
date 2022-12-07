	SUBROUTINE DO_AUX_VAR_REGRID_1D  ( axis, aux_has_cell_pts,
     .				           src,  msrc, src_cx,
     .				           dst,  mdst, dst_cx,
     .				           pos,  mpos, pos_cx,
     .					   kmin_pos, kmax_pos, 
     .					   kmin_src, kmax_src, 
     .					   kmin_dst, kmax_dst    )

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
* perform a 1D axis regrid (not just Z axis) guided by an auxiliary variable
* containing coord positions
* allow for the usual Ferret promotion of point axes to match span axes
* where either the source data or the position field may have degenerate axes
* (typically the position array will be degenerate in T, E and F) 
* numerical code lifted from do_internal_gc_fcn.F, ZAXREPLACE
* When comparing to that code
*      dst == res
*      src == com1
*      pos == com2
* Note that "k" that is found in the name of some variables is a meaningless
* hold-over from the original K-only version of this routine


* v690 11/13-4/14 *sh*
* V693+ 11/14 *sh* renaming 'sigma' as 'layerz' throughout
* v695+ 3/15 *sh*  change to call name to accommodate T axis regridding
* v698+ 5/15 *sh* linear regrid allows monotonic, but averaging doesnt
*       6/15 *sh* bug fix in determination of ascending affects
*                 false_1pt_regrid [gz(aux)=pt]
* V700 11/16 *sh* fixed bug in koffset_false lingering from K-only regrid
*                 days of this routine
* V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management 
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
        include 'tmap_dims.parm'
        include 'errmsg.parm'
	include	'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	
* calling argument declarations:
	LOGICAL	aux_has_cell_pts
	INTEGER	axis,
     .		msrc, mdst, mpos, src_cx, dst_cx, pos_cx,
     .		kmin_src, kmax_src, kmin_dst, kmax_dst, kmin_pos, kmax_pos
        REAL    src( m4lox:m4hix,m4loy:m4hiy,m4loz:m4hiz,
     .               m4lot:m4hit,m4loe:m4hie,m4lof:m4hif ),
     .          dst( m5lox:m5hix,m5loy:m5hiy,m5loz:m5hiz,
     .               m5lot:m5hit,m5loe:m5hie,m5lof:m5hif ),
     .          pos( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif )


* internal variable declarations:
	LOGICAL	new_aux_line, has_valid, ascending, false_1pt_regrid,
     .		monotonic_block
        INTEGER CX_DIM_LEN,
     .          idim,
     .          i,   j,   k,   l,   m,   n,
     .          is,  js,  ks,  ls,  ms,  ns,
     .          ip,  jp,  kp,  lp,  mp,  np,
     .          si0, sj0, sk0, sl0, sm0, sn0,
     .          pi0, pj0, pk0, pl0, pm0, pn0,
     .          dsi, dsj, dsk, dsl, dsm, dsn,
     .          dpi, dpj, dpk, dpl, dpm, dpn
        INTEGER ndx_lo, wkblk, kdmax, ss, trans, klook, khiok,
     .		koffset_false, kstart, kend, n_ascnd, n_desnd
        REAL    bad_src, bad_dst, bad_pos, zval, v0, v1
        REAL*8  TM_WORLD
* internally declared work arrays
	INTEGER	indices(kmin_dst:kmax_dst+1),
     .		dstlo(nferdims), dsthi(nferdims)
	REAL	  posline(kmin_pos:kmax_pos),srcline(kmin_src:kmax_src),
     .		  dstline(kmin_dst:kmax_dst),
     .		dstcoords(kmin_dst:kmax_dst+1),coef(kmin_dst:kmax_dst+1)


* equivalence conveniences
* "inc" is the increments for each axis of each component
* "lo"  is the starting subscript (minus 1) for each axis of each component
* "s" is the source data (src)
* "p" is the position data (pos) 
* "d" indicates the delta increment
* "0" indicates the start (lo) index
        INTEGER srcinc(nferdims), srclo(nferdims),
     .		posinc(nferdims), poslo(nferdims)
        EQUIVALENCE
     .	    (srcinc(1),dsi), (srcinc(2),dsj), (srcinc(3),dsk),
     .	    (srcinc(4),dsl), (srcinc(5),dsm), (srcinc(6),dsn),
     .	    (posinc(1),dpi), (posinc(2),dpj), (posinc(3),dpk),
     .	    (posinc(4),dpl), (posinc(5),dpm), (posinc(6),dpn),
     .	    (srclo(1), si0), (srclo(2), sj0), (srclo(3), sk0),
     .	    (srclo(4), sl0), (srclo(5), sm0), (srclo(6), sn0),
     .	    (poslo(1), pi0), (poslo(2), pj0), (poslo(3), pk0),
     .	    (poslo(4), pl0), (poslo(5), pm0), (poslo(6), pn0)

*==========================

* initialize
	kdmax = kmax_dst
	false_1pt_regrid = cx_aux_stat(axis,dst_cx)
     .		      .EQ. paux_stat_false_1pt_regrid
*   koffset_false fixes ss of -999 used as a valid position in the dest axis
* ==> "k" in names is a misleading relic of the Z-only origins of this routine
!	koffset_false = kmin_dst - m5loz   ! old Z-axis specific logic
	koffset_false = kmin_dst - lo_ss(5,axis)
	trans = cx_regrid_trans(axis,dst_cx)

* flag(s) for bad or missing values
        bad_src = mr_bad_data( msrc )
        bad_dst = mr_bad_data( mdst )
        bad_pos = mr_bad_data( mpos )

* compute the delta increment for each axis
* this allows "promotion" of degenerate axes
        DO idim = 1, nferdims
	   dstlo(idim) = mr_lo_ss(mdst,idim)
	   dsthi(idim) = mr_hi_ss(mdst,idim)
           IF ( CX_DIM_LEN(idim,src_cx) .EQ. 1 ) THEN
              srcinc(idim) = 0
           ELSE
              srcinc(idim) = 1
           ENDIF
           IF ( CX_DIM_LEN(idim,pos_cx) .EQ. 1 ) THEN
              posinc(idim) = 0
           ELSE
              posinc(idim) = 1
           ENDIF
	ENDDO
* ... a dummy loop on the designated axis 
	dstlo(axis)  = 1
	dsthi(axis)  = 1
	srcinc(axis) = 0
	posinc(axis) = 0

* compute the starting subscript for each axis of each component
* (pre-decrement by 1 delta for looping ahead)
* NOTE: 
        DO idim = 1, nferdims
	   srclo(idim) = cx_lo_ss(src_cx,idim) - srcinc(idim)
	   poslo(idim) = cx_lo_ss(pos_cx,idim) - posinc(idim)
	ENDDO

* extract the list of target coordinates
	IF (trans .EQ. pauxrgrd_linear ) THEN
	   IF (false_1pt_regrid) THEN
* ... false 1pt regridding - e.g. Z=100
	      dstcoords(kmin_dst) = cx_lo_ww(axis, dst_cx)
	   ELSE
	      DO k = kmin_dst, kmax_dst
                 dstcoords(k) = TM_WORLD(k,cx_grid(dst_cx),axis,box_middle)
	      ENDDO
	   ENDIF
	ELSEIF (trans .EQ. pauxrgrd_pwlave ) THEN
	   IF (false_1pt_regrid) THEN
* ... false 1pt regridding - e.g. Z=100:200@ave
	      dstcoords(kmin_dst  ) = cx_lo_ww(axis, dst_cx)
	      dstcoords(kmin_dst+1) = cx_hi_ww(axis, dst_cx)
	   ELSE
	      DO k = kmin_dst, kmax_dst
                 dstcoords(k) = TM_WORLD(k,cx_grid(dst_cx),axis,box_lo_lim)
	      ENDDO
	      dstcoords(kmax_dst+1) =
     .		    TM_WORLD(kmax_dst,cx_grid(dst_cx),axis,box_hi_lim)
	   ENDIF
	   kdmax = kdmax + 1
	ELSE
	   STOP 'do_aux_var_regrid_trns'   ! checked in AUX_REGRID_LIMS
	ENDIF

* initial code does only the Z axis, based upon "ZAXREPLACE(V,ZVALS,ZAX)"
* of old ... regrid a layer-Z-coordinate variable, V (src), onto
* a depth axis, ZAX, (Z axis of dst) guided by the depths in ZVALS (pos)


* loop over the X,Y,T,E,F range of the input fields
	new_aux_line = .TRUE.
	is = si0
	ip = pi0
	DO i = dstlo(x_dim), dsthi(x_dim)
	 is = is + dsi
	 IF (dpi .NE. 0) THEN
	   ip = ip + dpi
	   new_aux_line = .TRUE.
	 ENDIF
	 js = sj0
	 jp = pj0
	 DO j = dstlo(y_dim), dsthi(y_dim)
	  js = js + dsj
	  IF (dpj .NE. 0) THEN
	    jp = jp + dpj
	    new_aux_line = .TRUE.
	  ENDIF
	   ks = sk0
	   kp = pk0
	   DO k = dstlo(z_dim), dsthi(z_dim)
	    ks = ks + dsk
	    IF (dpk .NE. 0) THEN
	      kp = kp + dpk
	      new_aux_line = .TRUE.
	    ENDIF
	    ls = sl0
	    lp = pl0
	    DO l = dstlo(t_dim), dsthi(t_dim)
	     ls = ls + dsl
	     IF (dpl .NE. 0) THEN
	       lp = lp + dpl
	       new_aux_line = .TRUE.
	     ENDIF
	     ms = sm0
	     mp = pm0
	     DO m = dstlo(e_dim), dsthi(e_dim)
	      ms = ms + dsm
	      IF (dpm .NE. 0) THEN
	        mp = mp + dpm
	        new_aux_line = .TRUE.
	      ENDIF
	      ns = sn0
	      np = pn0
	      DO n = dstlo(f_dim), dsthi(f_dim)
	       ns = ns + dsn
	       IF (dpn .NE. 0) THEN
	         np = np + dpn
	         new_aux_line = .TRUE.
	       ENDIF

* compute on the source and destination coords only if they differ from last
	     IF (new_aux_line) THEN
* extract the line of coord vals from the aux variable at this location
* 6/2015-- do full region so that we get a reliable determination of ascending vs descending
	        CALL EXTRACT_ARG1_LINE( pos, axis, kmin_pos, kmax_pos,
     .				        ip, jp, kp, lp, mp, np, posline )


* 5/15 - allow for non-monotonic position array.
* PROGRAMMER NOTE: The 5/15 changes make this code seriously sub-optimal
* The code immediately following could be blended with the routines
* HUNT_INDICES and FULL_SCAN_INDICES to make something both smaller and faster

* check for a single contiguous block of monotonic points
	        monotonic_block = .FALSE.
	        n_ascnd = 0
	        n_desnd = 0
	        kstart = kmax_src
	        kend   = kmin_src

* ... count ascending and descending pairs
		v0 = posline(kmin_pos)
	        DO ss = kmin_pos+1, kmax_pos
		   v1 = posline(ss)
	           IF (v1 .NE. bad_pos) THEN
	              IF (v1 .GT. v0) THEN
	                 n_ascnd = n_ascnd + 1
	              ELSEIF (v1 .LT. v0) THEN
	                 n_desnd = n_desnd + 1
	              ENDIF
	              v0 = v1
	           ENDIF
	        ENDDO

* ... assess the monotonicity
*     Note: through this approach columns may differ in direction in pathological data!!
	        ascending = n_ascnd .GE. n_desnd

* ... find start of non-bad block
	        DO ss = kmin_src, kmax_src
	           IF (posline(ss) .NE. bad_pos) THEN
	              kstart = ss
	             EXIT
	           ENDIF
	        ENDDO
* ... find end of non-bad block
	        DO ss = kmax_src, kstart, -1
	           IF (posline(ss) .NE. bad_pos) THEN
	              kend = ss
	              EXIT
	           ENDIF
	        ENDDO

* ... no non-bad-block exists
	        CONTINUE

		CALL FULL_SCAN_INDICES(posline(kmin_src), kmin_src, kmax_src,
     .				  kstart, kend,
     .				  bad_pos, dstcoords,
     .				  indices, coef, kmin_dst, kdmax,
     .				  ascending, has_valid )
	        klook = kmin_dst
	        khiok = kdmax
	        IF (trans .EQ. pauxrgrd_pwlave ) khiok = khiok-1
	        new_aux_line = .FALSE.
	     ENDIF

	     IF (has_valid) THEN
* extract the line of source vals 
	        CALL EXTRACT_ARG4_LINE( src, axis, kmin_src, kmax_src,
     .					is, js, ks, ls, ms, ns, srcline )

* perform the regridding
	        IF (trans .EQ. pauxrgrd_linear ) THEN
* ... by linear interpolation to target cell coordinate points
	           CALL DO_AUX_REGRID_LINE_LIN (axis,
     .				  srcline,
     .				  kmin_src, kmax_src,
     .				  indices,  coef,
     .				  kmin_dst, kmax_dst,
     .				  klook,    khiok,
     .				  bad_src,  bad_dst, ascending,
     .				  dstline             )
	        ELSEIF (trans .EQ. pauxrgrd_pwlave ) THEN
* ... by within-target-cell averaging   (count non-monotonic positions)
	           CALL DO_AUX_REGRID_LINE_AVE (axis,
     .				  srcline,  posline(kmin_src),
     .				  kmin_src, kmax_src,
     .				  indices,  coef,
     .				  kmin_dst, kmax_dst,
     .				  klook,    khiok,
     .				  bad_src,  bad_dst, bad_pos,
     .				  ascending,
     .				  dstline             )
	        ENDIF

* put the line of results into the destination array
	        CALL PUT_ARG5_LINE(
     .			dst, axis, klook, khiok, koffset_false,
     .			i, j, k, l, m, n, dstline )

	     ENDIF

* Close nested i, j, k, l, m, n loops

	      ENDDO
	     ENDDO
	    ENDDO
	   ENDDO
	 ENDDO
	ENDDO

* successful completion.  Clean up.
        CALL RELEASE_DYN_WORK_SPACE
        RETURN

* error exit
 5000	RETURN
	END
