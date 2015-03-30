	SUBROUTINE DO_AUX_VAR_REGRID_1D_Z( axis, aux_has_cell_pts,
     .				           src,  msrc, src_cx,
     .				           dst,  mdst, dst_cx,
     .				           pos,  mpos, pos_cx,
     .					   kmin_src, kmax_src, 
     .					   kmin_dst, kmax_dst ) 

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
* perform a 1D Z axis regrid guided by an auxiliary variable
* containing coord positions
* allow for the usual Ferret promotion of point axes to match span axes
* where either the source data or the position field may have degenerate axes
* (typically the position array will be degenerate in T, E and F) 
* numerical code lifted from do_internal_gc_fcn.F, ZAXREPLACE
* When comparing to that code
*      dst == res
*      src == com1
*      pos == com2


* v690 11/13-4/14 *sh*
* V693+ 11/14 *sh* renaming 'sigma' as 'layerz' throughout
* v695+ 3/15 *sh*  change to call name to accommodate T axis regridding

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
     .		kmin_src, kmax_src, kmin_dst, kmax_dst
        REAL    src( m4lox:m4hix,m4loy:m4hiy,m4loz:m4hiz,
     .               m4lot:m4hit,m4loe:m4hie,m4lof:m4hif ),
     .          dst( m5lox:m5hix,m5loy:m5hiy,m5loz:m5hiz,
     .               m5lot:m5hit,m5loe:m5hie,m5lof:m5hif ),
     .          pos( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif )


* internal variable declarations:
	LOGICAL	new_aux_line, has_valid, ascending, false_1pt_regrid
        INTEGER CX_DIM_LEN,
     .          idim,
     .          i,   j,   k,   l,   m,   n,
     .          is,  js,  ks,  ls,  ms,  ns,
     .          ip,  jp,  kp,  lp,  mp,  np,
     .          si0, sj0, sk0, sl0, sm0, sn0,
     .          pi0, pj0, pk0, pl0, pm0, pn0,
     .          dsi, dsj, dsk, dsl, dsm, dsn,
     .          dpi, dpj, dpk, dpl, dpm, dpn
        INTEGER ndx_lo, wkblk, kdmax, ii, trans, klook, khiok,
     .		koffset_false
        REAL    bad_src, bad_dst, bad_pos, zval, frac
        REAL*8  TM_WORLD
* internally declared work arrays
	INTEGER	indices(kmin_dst:kmax_dst+1)
	REAL	  posline(kmin_src:kmax_src),srcline(kmin_src:kmax_src),
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
*   koofset_false fixes ss of -999 used as a valid position in the dest axis.
	koffset_false = kmin_dst - m5loz
	trans = cx_regrid_trans(axis,dst_cx) !  TEMPORARY: 1 dim HARD CODED!!!

* flag(s) for bad or missing values
        bad_src = mr_bad_data( msrc )
        bad_dst = mr_bad_data( mdst )
        bad_pos = mr_bad_data( mpos )

* compute the delta increment for each axis
* this allows "promotion" of degenerate axes
        DO 10 idim = 1, nferdims
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
 10     CONTINUE

* compute the starting subscript for each axis of each component
* (pre-decrement by 1 delta for looping ahead)
* NOTE: 
        DO 20 idim = 1, nferdims
	   srclo(idim) = cx_lo_ss(src_cx,idim) - srcinc(idim)
	   poslo(idim) = cx_lo_ss(pos_cx,idim) - posinc(idim)
 20	CONTINUE

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
	DO 1000 i = mr_lo_s1(mdst), mr_hi_s1(mdst)
	 is = is + dsi
	 IF (dpi .NE. 0) THEN
	   ip = ip + dpi
	   new_aux_line = .TRUE.
	 ENDIF
	 js = sj0
	 jp = pj0
	 DO 1000 j = mr_lo_s2(mdst), mr_hi_s2(mdst)
	  js = js + dsj
	  IF (dpj .NE. 0) THEN
	    jp = jp + dpj
	    new_aux_line = .TRUE.
	  ENDIF
	  ls = sl0
	  lp = pl0
!   ... k is the axis of the EXTRACT ...
	  DO 1000 l = mr_lo_s4(mdst), mr_hi_s4(mdst)
	   ls = ls + dsl
	   IF (dpl .NE. 0) THEN
	     lp = lp + dpl
	     new_aux_line = .TRUE.
	   ENDIF
	   ms = sm0
	   mp = pm0
	   DO 1000 m = mr_lo_s5(mdst), mr_hi_s5(mdst)
	    ms = ms + dsm
	    IF (dpm .NE. 0) THEN
	      mp = mp + dpm
	      new_aux_line = .TRUE.
	    ENDIF
	    ns = sn0
	    np = pn0
	    DO 1000 n = mr_lo_s6(mdst), mr_hi_s6(mdst)
	     ns = ns + dsn
	     IF (dpn .NE. 0) THEN
	       np = np + dpn
	       new_aux_line = .TRUE.
	     ENDIF


* compute on the source and destination coords only if they differ from last
	     IF (new_aux_line) THEN
* extract the line of source coord vals from the aux variable at this location
	        DO 500 k = kmin_src, kmax_src
 500		posline(k) = pos(ip, jp, k, lp, mp, np)

* for each destination point, find the source index just below this value
* and the weight coefficient to attach to that index 
* ... note that the indices are referenced to a starting index of 1
	        CALL HUNT_INDICES(posline, kmin_src, kmax_src,
     .				  bad_pos, dstcoords,
     .				  indices, coef, kmin_dst, kdmax,
     .				  klook, khiok, ascending )
	        has_valid = klook .NE. unspecified_int4
		IF (trans .EQ. pauxrgrd_pwlave ) khiok = khiok-1
		new_aux_line = .FALSE.
	     ENDIF

	     IF (has_valid) THEN
* extract the line of source vals 
	        DO 600 k = kmin_src, kmax_src
 600	        srcline(k) = src(is, js, k, ls, ms, ns)

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
* ... by within-target-cell averaging
	           CALL DO_AUX_REGRID_LINE_AVE (axis,
     .				  srcline,  posline,
     .				  kmin_src, kmax_src,
     .				  indices,  coef,
     .				  kmin_dst, kmax_dst,
     .				  klook,    khiok,
     .				  bad_src,  bad_dst, ascending,
     .				  dstline             )
	        ENDIF

* put the line of results into the destination array
	        DO k = klook, khiok
	          dst(i, j, k-koffset_false, l, m, n) = dstline(k)
	        ENDDO
	     ENDIF

 1000	CONTINUE

* successful completion.  Clean up.
        CALL RELEASE_WORK_SPC
        RETURN

* error exit
 5000	RETURN
	END
