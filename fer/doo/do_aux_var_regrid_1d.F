	SUBROUTINE DO_AUX_VAR_REGRID_1D( axis,
     .				         src,  msrc, src_cx,
     .				         dst,  mdst, dst_cx,
     .				         pos,  mpos, pos_cx,
     .					 work )

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
* perform a 1D regrid guided by an auxiliary variable containing coord positions
* allow for the usual Ferret promotion of point axes to match span axes
* where either the source data or the position field may have degenerate axes
* (typically the position array will be degenerate in T, E and F) 
* numerical code lifted from do_internal_gc_fcn.F, ZAXREPLACE
* When comparing to that code
*      dst == res
*      src == com1
*      pos == com2


* v685 4/13 *sh*

        IMPLICIT NONE
        include 'tmap_dims.parm'
        include 'errmsg.parm'
	include	'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'

* calling argument declarations:
	INTEGER	axis,
     .		msrc, mdst, mpos, src_cx, dst_cx, pos_cx
        REAL    src( m4lox:m4hix,m4loy:m4hiy,m4loz:m4hiz,
     .               m4lot:m4hit,m4loe:m4hie,m4lof:m4hif ),
     .          dst( m5lox:m5hix,m5loy:m5hiy,m5loz:m5hiz,
     .               m5lot:m5hit,m5loe:m5hie,m5lof:m5hif ),
     .          pos( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m2loe:m1hie,m1lof:m1hif ),
     .		work(*)


* internal variable declarations:
        INTEGER CX_DIM_LEN,
     .          idim,
     .          i,   j,   k,   l,   m,   n,
     .          is,  js,  ks,  ls,  ms,  ns,
     .          ip,  jp,  kp,  lp,  mp,  np,
     .          si0, sj0, sk0, sl0, sm0, sn0,
     .          pi0, pj0, pk0, pl0, pm0, pn0,
     .          dsi, dsj, dsk, dsl, dsm, dsn,
     .          dpi, dpj, dpk, dpl, dpm, dpn
        INTEGER klo, khi, ndx_lo, wkblk
        REAL    bad_src, bad_dst, bad_pos, zval, frac
        REAL*8  TM_WORLD

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

* initial code does only the Z axis, based upon "ZAXREPLACE(V,ZVALS,ZAX)"
* of old ... regrid a sigma-coordinate-style (layered) variable, V (src), onto
* a depth axis, ZAX, (Z axis of dst) guided by the depths in ZVALS (pos)


***************************************
*   ** to Ansley **
* ==> Note 1 The current code handles the Z axis regridding, only.  It needs to be
* generalized to allow the same operation over any axis.
* 
* ==> NOTE 2 The looping logic could be optimized to recognize that it is very 
*  rare that the position array has T, E, orF axes. In fact, it would be acceptable
* to forbid this case. This would allow EXTRACT_Z_LINE to be called firstjust once
* for all of the looping over T, E, and F
****************************************

* ... loop over the X,Y,T,E,F range of the input fields
	ns = sn0
	np = pn0
	DO 1450 n = mr_lo_s6(mdst), mr_hi_s6(mdst)
	 ns = ns + dsn
	 np = np + dpn
	 ms = sm0
	 mp = pm0
	 DO 1450 m = mr_lo_s5(mdst), mr_hi_s5(mdst)
	  ms = ms + dsm
	  mp = mp + dpm
	  ls = sl0
	  lp = pl0
	  DO 1450 l = mr_lo_s4(mdst), mr_hi_s4(mdst)
	   ls = ls + dsl
	   lp = lp + dpl
	   js = sj0
	   jp = pj0
	   DO 1450 j = mr_lo_s2(mdst), mr_hi_s2(mdst)
	    js = js + dsj
	    jp = jp + dpj
	    is = si0
	    ip = pi0
	    DO 1450 i = mr_lo_s1(mdst), mr_hi_s1(mdst)
	     is = is + dsi
	     ip = ip + dpi

* ... extract the vertical profile at this I,J,L location
             CALL EXTRACT_Z_LINE ( pos, bad_pos, ip, jp, lp, mp, np,
     .               cx_lo_s3(pos_cx), cx_hi_s3(pos_cx), 
     .               m1lox, m1hix, m1loy, m1hiy, m1loz, m1hiz,
     .               m1lot, m1hit, m1loe, m1hie, m1lof, m1hif,
     .               work, klo, khi )

             IF ( klo .GE. 1 ) THEN

* loop over the desired output Z-axis points
                ndx_lo = 1    ! initial guess
	        DO 1410 k = mr_lo_s3(mdst), mr_hi_s3(mdst)  ! destination Z axis

* ... next Z axis value (com3) to locate
#ifdef double_p
                   zval = TM_WORLD(k,cx_grid(dst_cx),z_dim,box_middle)
#else
                   zval = SNGL(TM_WORLD(k,cx_grid(dst_cx),z_dim,box_middle))
#endif
* ... search the position field for the index just below this value
* ... note that ndx_lo is always referenced to a starting index of 1
                   CALL HUNT_R4(work, klo, khi, zval, ndx_lo, frac)
                   IF ( frac .LT. 0.0 ) THEN
* ... no luck -- the Z axis value we seek is outside the range in com2
                      dst(i,j,k,l,m,n) = bad_dst
                   ELSEIF ( src(is,js,sk0+ndx_lo,ls,ms,ns)
     .                      .EQ. bad_src ) THEN
                      dst(i,j,k,l,m,n) = bad_dst
                   ELSEIF ( frac .EQ. 1.0 ) THEN                  ! exactly on point
                      dst(i,j,k,l,m,n) =
     .                   src(is,js,sk0+ndx_lo,ls,ms,ns)
                   ELSEIF ( src(is,js,sk0+ndx_lo+1,ls,ms,ns)
     .                      .EQ. bad_src ) THEN
                      dst(i,j,k,l,m,n) = bad_dst
                   ELSE
* ... interpolate data com1 field to get result
                      dst(i,j,k,l,m,n) = 
     .                   frac  * src(is,js,sk0+ndx_lo,  ls,ms,ns)
     .              + (1-frac) * src(is,js,sk0+ndx_lo+1,ls,ms,ns)
                   ENDIF
 1410           CONTINUE

             ENDIF

 1450	CONTINUE
        CALL RELEASE_WORK_SPC
        RETURN

* error exit
 5000	RETURN
	END
