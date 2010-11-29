	SUBROUTINE DD1STEP (	idim, shf, com_lo, com_hi, box,
     .				com, com_mr,
     .				res, res_mr, res_cx )

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
* compute a 1 step derivative of the component along axis idim
* note: for a forward  derivative shf=0, and com limits are res_lo:res_hi+1
*       for a backward derivative shf=1, and com limits are res_lo-1:res_hi

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/6/89
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN & com_lo/hi
* V230: 5/6/92 - COS correction checks that J location is defined

#ifdef unix
	include	'ferret.parm'
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
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* calling argument declarations:
	INTEGER	idim, shf, com_mr, com_lo, com_hi, res_mr, res_cx
	REAL	box( com_lo:com_hi ),
     .		com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit )

* internal variable declarations:
	LOGICAL	GEOG_COS_FACTOR, need_cos
	INTEGER i, j, k, l, ii, grid, lo_sub(4), hi_sub(4), axis
	REAL	bad_com, bad_res, lo_com, hi_com, unit, cos_factor,
     .          ybxlo, ybxhi, bx
	REAL*8	TM_WORLD, lo_ww, hi_ww

* internal (convenience) equivalences
	INTEGER	lo_s1, lo_s2, lo_s3, lo_s4, hi_s1, hi_s2, hi_s3, hi_s4
	EQUIVALENCE (lo_s1, lo_sub(1)) , (lo_s2, lo_sub(2)),
     .		    (lo_s3, lo_sub(3)) , (lo_s4, lo_sub(4)),
     .		    (hi_s1, hi_sub(1)) , (hi_s2, hi_sub(2)),
     .		    (hi_s3, hi_sub(3)) , (hi_s4, hi_sub(4))

* initialize
	grid    = mr_grid( res_mr )
	axis    = grid_line( idim, grid )
	bad_com = mr_bad_data( res_mr )
	bad_res = mr_bad_data( com_mr )

* compute limits within which calculation is possible
	DO 10 ii = 1, 4
	   lo_sub(ii) = cx_lo_ss(res_cx,ii)
	   hi_sub(ii) = cx_hi_ss(res_cx,ii)
 10	CONTINUE
	lo_sub(idim) = ( com_lo     ) + shf
	hi_sub(idim) = ( com_hi - 1 ) + shf

* create an array of delta H increments
	lo_ww = TM_WORLD( lo_sub(idim)-shf, grid, idim, box_middle )
	DO 20 ii = lo_sub(idim), hi_sub(idim)
	   hi_ww = TM_WORLD( ii+1-shf, grid, idim, box_middle )
	   box(ii) = hi_ww - lo_ww
	   lo_ww = hi_ww
 20	CONTINUE	   

* perform differentiation in standard units if possible
	unit = un_convert( line_unit_code(axis) )
	DO 30 ii = lo_sub(idim), hi_sub(idim)
 30	box(ii) = box(ii) * unit

* DERIVATIVE ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   need_cos = GEOG_COS_FACTOR( idim, grid )
     .                .AND. lo_s2.NE.unspecified_int4
	   DO 110 j = lo_s2, hi_s2
	   IF ( need_cos ) THEN
	      cos_factor = COS( deg2rad *       
     .             SNGL( TM_WORLD( j, grid, y_dim, box_middle )) ) 
              ybxlo = deg2rad* 
     .          SNGL( TM_WORLD(j,grid,y_dim,box_lo_lim) )
              ybxhi = deg2rad* 
     .          SNGL( TM_WORLD(j,grid,y_dim,box_hi_lim) )
              bx = ABS(ybxhi - ybxlo)
	      cos_factor = ABS( SIN(ybxhi) - SIN(ybxlo) )/ bx
	   ENDIF

	   DO 110 l = lo_s4, hi_s4
	   DO 110 k = lo_s3, hi_s3
	      lo_com = com(lo_s1-shf,j,k,l)
	      DO 100 i = lo_s1, hi_s1
	         hi_com = com(i+1-shf,j,k,l)
	         IF ( lo_com.EQ.bad_res .OR. hi_com.EQ.bad_res ) THEN
	            res(i,j,k,l) = bad_res
	         ELSE
	            res(i,j,k,l) = (hi_com-lo_com) / box(i)
	            IF ( need_cos ) res(i,j,k,l) = res(i,j,k,l)/cos_factor
	         ENDIF
	         lo_com = hi_com
 100	      CONTINUE
 110	   CONTINUE

* DERIVATIVE ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO 210 l = lo_s4, hi_s4
	   DO 210 k = lo_s3, hi_s3
	   DO 210 i = lo_s1, hi_s1
	      lo_com = com(i,lo_s2-shf,k,l)
	      DO 200 j = lo_s2, hi_s2
	         hi_com = com(i,j+1-shf,k,l)
	         IF ( lo_com.EQ.bad_res .OR. hi_com.EQ.bad_res ) THEN
	            res(i,j,k,l) = bad_res
	         ELSE
	            res(i,j,k,l) = (hi_com-lo_com) / box(j)
	         ENDIF
	         lo_com = hi_com
 200	      CONTINUE
 210	   CONTINUE

* DERIVATIVE ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 310 l = lo_s4, hi_s4
	   DO 310 j = lo_s2, hi_s2
	   DO 310 i = lo_s1, hi_s1
	      lo_com = com(i,j,lo_s3-shf,l)
	      DO 300 k = lo_s3, hi_s3
	         hi_com = com(i,j,k+1-shf,l)
	         IF ( lo_com.EQ.bad_res .OR. hi_com.EQ.bad_res ) THEN
	            res(i,j,k,l) = bad_res
	         ELSE
	            res(i,j,k,l) = (hi_com-lo_com) / box(k)
	         ENDIF
	         lo_com = hi_com
 300	      CONTINUE
 310	   CONTINUE

* DERIVATIVE ALONG T AXIS
	ELSE
	   DO 410 k = lo_s3, hi_s3
	   DO 410 j = lo_s2, hi_s2
	   DO 410 i = lo_s1, hi_s1
	      lo_com = com(i,j,k,lo_s4-shf)
	      DO 400 l = lo_s4, hi_s4
	         hi_com = com(i,j,k,l+1-shf)
	         IF ( lo_com.EQ.bad_res .OR. hi_com.EQ.bad_res ) THEN
	            res(i,j,k,l) = bad_res
	         ELSE
	            res(i,j,k,l) = (hi_com-lo_com) / box(l)
	         ENDIF
	         lo_com = hi_com
 400	      CONTINUE
 410	   CONTINUE

	ENDIF

	RETURN
	END	
